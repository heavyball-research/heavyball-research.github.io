---
title: "Flywheel Attention: Attention Kernels Built for the TPU"
wrap_tables: true
authors:
  - name: Zishuo Bao
    url: https://twitter.com/David_Bao031126
  - name: Zeshen Zhang
    url: https://twitter.com/Zeshen_Zhang
  - name: Linkun Chen
    url: https://github.com/lk-chen
  - name: Yucheng Lu
    url: https://twitter.com/_yucheng_lu
tldr: "TPU-native attention kernels on v6e and v7x: up to 4.14× faster softmax attention than Ragged Paged Attention (RPA-v3) and 1.80× faster than Splash Attention, reaching 81% of peak BF16 throughput on v6e; up to 3.61× faster Gated DeltaNet prefill and 1.56× higher prefill throughput in vLLM serving Qwen3.8-27B, with comparable accuracy."
image: /assets/posts/2026-09-20-attention-tpu/paper/gdn_kernel.png
thumbnail_large: true
links:
  - label: Code
    url: https://github.com/heavyball-research/flywheel-tpu
---

> Editor's Note: Updated in October 2026 with figures and results from our paper, including TPU v7x (Ironwood). The [code and benchmarks](https://github.com/heavyball-research/flywheel-tpu) are open source.

<nav class="post-toc" aria-labelledby="contents-title" markdown="1">
<p id="contents-title"><strong>Contents</strong></p>

* Contents
{:toc}

</nav>

Attention is all you need. But how much of the hardware can you actually use to run it?

On GPU, attention kernels are a well-studied problem. FlashAttention [[1]](#ref-1), [[2]](#ref-2), [[3]](#ref-3), [[4]](#ref-4) established the fused, tiled, online-softmax approach. FlashInfer [[6]](#ref-6) provides kernels for serving, and the FLA library [[7]](#ref-7) does the same for linear attention. Their success comes from designing around the machine: how it moves data, reuses operands, and overlaps matrix multiplication with other work.

On TPU, there is still a large gap between the hardware's capacity and what attention kernels achieve. In our causal prefill benchmarks on v6e, **RPA-v3**, the Ragged Paged Attention serving kernel, reaches 18–53% of roofline throughput. **Splash Attention**, Google's dense attention kernel, reaches 35–73%. Both already use blocked attention. So where does the rest of the performance go?

<details markdown="1" id="baseline-details">
<summary>Baselines</summary>

These are concrete TPU implementations, not different model architectures. The comparisons in this post use the versions pinned in the paper:

- **RPA-v3 (Ragged Paged Attention v3)** [[8]](#ref-8) is the softmax attention kernel in vLLM's `tpu-inference` backend. It reads a paged key/value cache and handles sequences of different lengths in the same batch, covering prefill and decode. We use [`tpu-inference` at commit `4420cae`](https://github.com/vllm-project/tpu-inference/tree/4420cae/tpu_inference/kernels/ragged_paged_attention/v3).
- **Splash Attention** [[29]](#ref-29) is the dense, blocked attention kernel maintained in OpenXLA's Tokamax and used by MaxText for training. It operates on dense sequences and supports block-sparse masks. We use [Tokamax at commit `84e5f36`](https://github.com/openxla/tokamax/tree/84e5f36/tokamax/_src/ops/experimental/tpu/splash_attention).
- **GDN-v3** [[30]](#ref-30) is the fused causal convolution and Gated DeltaNet prefill kernel in [`tpu-inference` at commit `4420cae`](https://github.com/vllm-project/tpu-inference/tree/4420cae/tpu_inference/kernels/gdn/v3). Gated DeltaNet is the attention mechanism; GDN-v3 is the baseline implementation of it.

The application comparison uses vLLM at commit `51da0ca` with `tpu-inference` at `4420cae`. An environment switch selects the kernels inside the same build. These measurements describe those versions, rather than whichever versions the repositories contain today.

</details>

The answer is in the work between the matmuls: loading operands, reducing and rescaling scores, and waiting for dependent operations. The GPU techniques above solve these problems around a different machine. On TPU, we have to revisit the layout, the arithmetic, and the schedule together.

**Flywheel Attention** applies that reasoning to both softmax attention and Gated DeltaNet. The result is up to 81% of peak BF16 throughput on v6e, softmax speedups of up to 4.14× over RPA-v3 and 1.80× over Splash across v6e and v7x, and up to 3.61× faster GDN prefill. In vLLM serving Qwen3.8-27B, the kernels improve prefill throughput by up to 1.56×. The rest of this post follows the three bottlenecks that led us there.

## Background

**Softmax attention** reads queries, keys, and values, computes attention weights, and uses those weights to mix the values. FlashAttention avoids building the full attention matrix in memory: it visits blocks of keys and values while keeping a running output. Each block contains two matrix products with softmax operations between them.

<details markdown="1" id="softmax-background">
<summary>Softmax Attention: Blockwise Computation</summary>

We use the paper's notation. For one head, $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{T\times D}$, where $T$ is the sequence length and $D$ the head dimension. Softmax attention computes

$$
\mathbf{P}=\operatorname{softmax}(\mathbf{Q}\mathbf{K}^{\top}),
\qquad \mathbf{O}=\mathbf{P}\mathbf{V}.
$$

Here $\mathbf{P}\in\mathbb{R}^{T\times T}$ and $\mathbf{O}\in\mathbb{R}^{T\times D}$. Scaling and masks are omitted for clarity. Multi-head attention (MHA) gives each query head its own key/value head; grouped-query attention (GQA) shares key/value heads across query heads. We write their counts as $H\_q$ and $H\_{kv}$.

FlashAttention [[1]](#ref-1) never materializes the full $\mathbf{P}$. For one query row, online softmax keeps a scalar running maximum $m$, a scalar normalization sum $l$, and an output accumulator $\mathbf{o}\in\mathbb{R}^{1\times D}$. For a block of $B$ keys, let $\mathbf{s}\in\mathbb{R}^{1\times B}$ be its scores and $\mathbf{V}\_b\in\mathbb{R}^{B\times D}$ its values. The update is

$$
\begin{aligned}
m' &= \max(m,\max_j s_j),\\
\mathbf{p} &= e^{\mathbf{s}-m'},\qquad r=e^{m-m'},\\
l &\leftarrow r l+\textstyle\sum_j p_j,\\
\mathbf{o} &\leftarrow r\mathbf{o}+\mathbf{p}\mathbf{V}_b.
\end{aligned}
$$

We then set $m\leftarrow m'$ and return $\mathbf{o}/l$ after the last block. The factor $r$ rescales earlier contributions whenever the running maximum increases. That stabilizes the exponentials, but adds reductions, rescaling, and dependencies between the matrix products.

</details>

**Gated DeltaNet (GDN)** [[10]](#ref-10) replaces the growing key/value cache with a fixed-size state. Each token updates that state and reads an output from it. For prefill, grouping tokens into chunks turns most of the recurrence into matrix products [[11]](#ref-11). The difficult operation in the middle is a triangular inverse. Before the recurrence, the layer also applies a short causal convolution, an activation, and query/key normalization.

<details markdown="1" id="gdn-background">
<summary>Gated DeltaNet: Chunked Computation</summary>

Following the paper, queries and keys are row vectors $\mathbf{q}\_t,\mathbf{k}\_t\in\mathbb{R}^{1\times d\_k}$, values are $\mathbf{v}\_t\in\mathbb{R}^{1\times d\_v}$, and the state is $\mathbf{S}\_t\in\mathbb{R}^{d\_k\times d\_v}$. The update and output are

$$
\begin{aligned}
\mathbf{S}_t &= \mathbf{S}_{t-1}
+\beta_t\mathbf{k}_t^\top(\mathbf{v}_t-\mathbf{k}_t\mathbf{S}_{t-1}),\\
\mathbf{o}_t &= \mathbf{q}_t\mathbf{S}_t.
\end{aligned}
$$

The scalar $\beta\_t$ controls the update strength. The residual in parentheses corrects the state's prediction of the current value. As in the paper's background, we omit the decay gate, which scales the previous state before the update.

Within a chunk of $C=64$ tokens, stack the rows into $\mathbf{Q},\mathbf{K}\in\mathbb{R}^{C\times d\_k}$ and $\mathbf{V}\in\mathbb{R}^{C\times d\_v}$. Let $\boldsymbol{\beta}=\operatorname{diag}(\beta\_1,\ldots,\beta\_C)$ and define

$$
\begin{aligned}
\mathbf{A}_{kk}&=\operatorname{tril}_{-1}(\boldsymbol{\beta}\mathbf{K}\mathbf{K}^\top),\\
\mathbf{A}_{qk}&=\operatorname{tril}_{0}(\mathbf{Q}\mathbf{K}^\top).
\end{aligned}
$$

Both matrices have shape $C\times C$. The first mask keeps the strictly lower triangle; the second includes the diagonal. With $\mathbf{S}$ the state entering the chunk,

$$
\begin{aligned}
\mathbf{V}_{\mathrm{new}} &=(\mathbf{I}+\mathbf{A}_{kk})^{-1}
\boldsymbol{\beta}(\mathbf{V}-\mathbf{K}\mathbf{S}),\\
\mathbf{O}&=\mathbf{Q}\mathbf{S}+\mathbf{A}_{qk}\mathbf{V}_{\mathrm{new}},\\
\mathbf{S}&\leftarrow\mathbf{S}+\mathbf{K}^\top\mathbf{V}_{\mathrm{new}}.
\end{aligned}
$$

Here $\mathbf{V}\_{\mathrm{new}},\mathbf{O}\in\mathbb{R}^{C\times d\_v}$ and $\mathbf{I}$ is the $C\times C$ identity. Decay factors are again omitted. The inputs here are after convolution, SiLU, and query/key normalization.

There are two useful kinds of independence in these equations. The local matrices and their inverse depend only on the chunk's inputs. The value updates, outputs, and state update also need the incoming recurrent state. That distinction will matter when we build the pipeline.

</details>

These mechanisms look different, but both alternate matrix products with work that can keep the matrix units waiting. That is the shared problem Flywheel addresses.

## TPU Architecture and Attention Bottlenecks

GPU attention optimizations do not transfer directly because the TPU changes the cost of operand reuse, vector work, and scheduling [[19]](#ref-19), [[20]](#ref-20), [[24]](#ref-24). The units can execute concurrently; the challenge is giving them useful work at the right time.

### Data Layout and Matrix Operand Reuse

A TPU matrix unit (MXU) is weight-stationary: it loads the right-hand operand into a systolic array, then reuses it as left-hand rows stream through. Loading that operand costs something even if the data is already on chip. A layout that leaves padding in the tile, or allows only a few query rows per load, wastes matrix capacity.

On GPUs, attention kernels organize loads and work across threads around GPU tensor-core instructions. TPU register tiling and the MXU's separate operand-loading step make a different organization useful. Optimizing just the matmul is insufficient if the surrounding model must first upcast or retile its inputs in high-bandwidth memory (HBM).

### Non-matrix Work on the Critical Path

TPUs have much more matrix throughput than vector throughput. Softmax's maximum reductions and accumulator rescales sit between its two matrix products. GDN's forward substitution introduces a chain of dependent row updates. Neither looks large in a FLOP count, but either can hold up the MXU.

Even a GPU optimization aimed at reducing this work can have a different outcome on TPU. FlashAttention-4 [[4]](#ref-4) conditionally skips rescaling. In our TPU implementation, it still needs the per-fragment maximum and rescale-factor computation, and its branch adds register pressure. Saving arithmetic alone does not guarantee a faster kernel.

### Exposing Work for Compiler Scheduling

FlashAttention-3 [[3]](#ref-3) combines asynchronous matrix operations with pipelining across GPU warpgroups. A TPU TensorCore instead coordinates its matrix, vector, and specialized units through one instruction stream. The compiler can overlap independent operations, but it cannot overlap a dependency chain with itself.

Within a softmax fragment, scores feed exponentials, which feed the output product. In GDN, the inverse has dependent products and the recurrent state links chunks. The independent work is in other fragments and chunks. The kernel must expose it across the boundaries that would otherwise hide it from the scheduler.

## Flywheel

The three challenges suggest three connected changes: make each operand load do more work, reduce the vector operations around the matmuls, and interleave independent computation. We apply each idea to both attention mechanisms.

### Layout and Operand Reuse

For softmax attention, the first choice is what to put on the TPU's register sublanes. RPA-v3 puts heads there. Every query block then carries all the heads of its tokens, together with their accumulators. The more heads on a chip, the shorter the blocks, and every extra block re-reads keys and values. Under MHA, padding also leaves half the BF16 rows unused.

We instead put **tokens on sublanes**, with one head per tile. Each tile is a dense matrix, its query length no longer shrinks with the head count, and each loaded key or value operand serves more query rows. GQA follows the same organization: each query head reads its shared key/value head. The surrounding projections produce and consume this layout directly, so the gain does not require a separate HBM conversion.

<figure class="figure-compact">
  <a href="{{ '/assets/posts/2026-09-20-attention-tpu/paper/layout.pdf' | relative_url }}"><img src="{{ '/assets/posts/2026-09-20-attention-tpu/paper/layout.png' | relative_url }}" alt="Per-head token tiling and on-chip handling of ragged sequence boundaries." width="1800" height="1462" loading="lazy"></a>
  <figcaption>Figure 1. Top: RPA-v3 places heads on sublanes; Flywheel packs tokens from one head, removing the MHA padding and allowing longer query tiles. Bottom: a sequence beginning at row 13 uses an aligned window starting at row 8. Rows 8–12 are restored from the previous VMEM buffer before the output DMA.</figcaption>
</figure>

GDN's chunks are smaller, so we find reuse across products instead. The products that build its two local matrices share the same keys. Later, the output contribution and state update share the newly computed values. **Stacking products with the same right operand turns four separate matmuls into two larger ones.** Each operand is loaded once for both uses.

<details markdown="1" id="operand-reuse">
<summary>GDN Shared-Operand Products</summary>

Using the chunk notation above,

$$
\begin{bmatrix}\boldsymbol{\beta}\mathbf{K}\\\mathbf{Q}\end{bmatrix}\mathbf{K}^\top
=
\begin{bmatrix}\boldsymbol{\beta}\mathbf{K}\mathbf{K}^\top\\\mathbf{Q}\mathbf{K}^\top\end{bmatrix}.
$$

After triangular masking, this gives $\mathbf{A}\_{kk}$ and $\mathbf{A}\_{qk}$ with one load of $\mathbf{K}^\top$. The state increment and intra-chunk output similarly share $\mathbf{V}\_{\mathrm{new}}$:

$$
\begin{bmatrix}\mathbf{K}^\top\\\mathbf{A}_{qk}\end{bmatrix}\mathbf{V}_{\mathrm{new}}
=
\begin{bmatrix}\mathbf{K}^\top\mathbf{V}_{\mathrm{new}}\\\mathbf{A}_{qk}\mathbf{V}_{\mathrm{new}}\end{bmatrix}.
$$

These products preserve the paper's state orientation: $\mathbf{S}\in\mathbb{R}^{d\_k\times d\_v}$, so the state increment is $\mathbf{K}^\top\mathbf{V}\_{\mathrm{new}}$.

</details>

The boundary around the kernel matters too. GDN-v3's wrapper upcasts and retiles the projection output. Flywheel takes the BF16 projection layout directly and writes the layout consumed by the next projection. In the paper's 8K-token ablation, coordinated layouts alone give a 1.52–1.68× speedup.

### Reducing Vector Work

Better reuse gives the MXU more useful work per load. Now consider what happens between those products.

**For softmax, stop repeatedly updating the anchor.** Online softmax rescales the whole accumulator whenever its running maximum increases. But softmax does not require that reference to be the maximum: subtracting any fixed value multiplies all the weights by a common factor, which cancels in the final normalization. The reference only has to keep the arithmetic within range.

Flywheel starts each row's anchor at the maximum of its first key tile and holds it fixed on the fast path. A later score could still be much larger. Rather than recompute a maximum on every fragment, we **check the sum already being accumulated**. The weights are nonnegative, so that sum bounds every individual weight. One comparison per row at the end of a key/value block checks the range budget. A failed block is replayed with guarded anchor updates in the same launch.

<details markdown="1" id="checked-anchor">
<summary>Checked Static Anchor and Guarded Replay</summary>

For a query row with scalar anchor $a$, scores $\mathbf{s}$, normalization sum $l$, and output accumulator $\mathbf{o}$, the fast path computes

$$
\begin{aligned}
\mathbf{p}&=e^{\mathbf{s}-a},\\
l&\leftarrow l+\textstyle\sum_j p_j,\\
\mathbf{o}&\leftarrow\mathbf{o}+\mathbf{p}\mathbf{V}_b.
\end{aligned}
$$

The final $\mathbf{o}/l$ cancels the common factor $e^{-a}$ in exact arithmetic. Unlike online softmax, no fragment updates the maximum or rescales the accumulators on this path.

The paper checks $l\le\Lambda$ after each key/value block. Since every $p\_j$ is nonnegative, $p\_j\le l$ for every key seen so far. A conservative sum therefore certifies the weights without inspecting the scores again. The comparison also fails for NaN and infinity.

Let $V\_{\max}$ bound the absolute entries of $\mathbf{V}$, and let $F$ be a conservative range limit supported by the operands and FP32 accumulators. For sequence length $T$, a sufficient condition for the guarded score margin $\tau\ge0$ is

$$
T e^\tau\max(1,V_{\max})\le F e^{-\delta},
$$

where $\delta\ge0$ reserves numerical headroom. A conservative fast-path acceptance bound is $\Lambda=e^\tau$. In exact arithmetic, an accepted sum bounds every output entry by $lV\_{\max}$. The implementation must also account for its operand casts and reductions.

For recovery, the kernel saves the anchor and accumulators on entry to each key/value block. If the check fails, it restores that state and recomputes **only the failing block**. The saved state includes all contributions before that block. Replay also completes the block's last output product and clears its pending weights, so the pipeline does not count that product twice. Before each fragment's exponentiation, the guarded path computes its maximum $m$ and chooses

$$
\begin{aligned}
a'&=\operatorname{where}(m-a>\tau,m,a),\\
r&=e^{a-a'},\\
l&\leftarrow r l,\qquad \mathbf{o}\leftarrow r\mathbf{o},\qquad a\leftarrow a'.
\end{aligned}
$$

Rescaling both accumulators preserves their ratio. After an anchor update, each score is at most $\tau$ above the anchor, giving the range bound above. The first key tile must contain a valid key for every query row; fully masked later fragments contribute zero. The softmax expression is unchanged in exact arithmetic, with ordinary floating-point error in the implementation.

Related fixed-anchor paths include FlashDecoding++ [[5]](#ref-5) and Splash Attention's caller-supplied `max_logit_const` [[29]](#ref-29). Flywheel chooses the anchor internally, checks its accumulated sum, and recovers inside the same launch, without offline calibration or a constant supplied by the caller.

</details>

In the paper's 32K-token v6e ablation, replacing online softmax with this checked anchor raises pipelined throughput from **582 to 722 TFLOP/s**. The gain comes from removing recurring work and its register state, not just skipping occasional rescales.

**For GDN, move the inverse onto the MXU.** GDN-v3 computes the triangular inverse with dependent forward-substitution updates on the vector unit. The inverse can also be expressed as matrix products, but applying a polynomial to the entire chunk can create large intermediate values and cancellation.

Flywheel uses a hierarchy: invert the **4×4** diagonal blocks, then merge four at a time, **4 → 16 → 64**. This builds on prior matrix-based inversion [[9]](#ref-9), with a decomposition designed for BF16 operands and FP32 accumulation. It completes the 64-token inverse in **eight matmuls**, avoiding high powers of the full-chunk matrix.

<figure class="figure-compact">
  <a href="{{ '/assets/posts/2026-09-20-attention-tpu/paper/vector.pdf' | relative_url }}"><img src="{{ '/assets/posts/2026-09-20-attention-tpu/paper/vector.png' | relative_url }}" alt="Hierarchical triangular inversion and causal convolution expressed as matrix products." width="1800" height="1441" loading="lazy"></a>
  <figcaption>Figure 2. Top: invert the 4×4 diagonal blocks, then merge four at a time, 4 → 16 → 64, using eight BF16 matmuls in total. Bottom: express the four-tap causal convolution as one product against a constant selector matrix; the preprocessing details are collected later in the post.</figcaption>
</figure>

<details markdown="1" id="hierarchical-inverse">
<summary>Hierarchical Triangular Inversion</summary>

Write the chunk's unit lower-triangular matrix as $\mathbf{U}=\mathbf{I}+\mathbf{A}\_{kk}$, as in the paper. In each 4×4 diagonal block, the strictly lower-triangular part $\mathbf{L}$ satisfies $\mathbf{L}^4=0$. Therefore

$$
(\mathbf{I}+\mathbf{L})^{-1}
=(\mathbf{I}-\mathbf{L})(\mathbf{I}+\mathbf{L}^2).
$$

This takes two products: one for $\mathbf{L}^2$, one for the factored inverse. The independent diagonal blocks are handled together.

At a merge level, decompose $\mathbf{U}=\mathbf{J}+\mathbf{E}$, where $\mathbf{J}$ contains four diagonal blocks and $\mathbf{E}$ their strictly lower off-diagonal blocks. The previous level already provides $\mathbf{J}^{-1}$. Define $\mathbf{N}=\mathbf{E}\mathbf{J}^{-1}$. It is strictly lower triangular in four-block form, so $\mathbf{N}^4=0$, and

$$
\mathbf{U}^{-1}
=\mathbf{J}^{-1}(\mathbf{I}-\mathbf{N})(\mathbf{I}+\mathbf{N}^2).
$$

Each merge takes three products:

1. Form $\mathbf{N}=\mathbf{E}\mathbf{J}^{-1}$.
2. Stack $\mathbf{J}^{-1}$ and $\mathbf{N}$ to form both $\mathbf{J}^{-1}\mathbf{N}$ and $\mathbf{N}^2$ with a shared right operand.
3. Multiply $(\mathbf{J}^{-1}-\mathbf{J}^{-1}\mathbf{N})$ by $(\mathbf{I}+\mathbf{N}^2)$.

Two products at the base and three at each of two merge levels give $2+3+3=8$. BF16 operands with FP32 accumulation produce an approximate inverse in floating-point arithmetic. On the paper's repeated-key test at $C=64$, its maximum absolute error relative to FP64 is $9\times10^{-3}$; the application results below evaluate model accuracy.

</details>

### Overlapping Independent Computation

Reducing the vector work still leaves latency. A softmax fragment has to compute its scores before its exponentials; the inverse has to wait for one product before issuing the next. Running each chain straight through leaves the MXU idle between dependent steps.

The useful independence is just outside each chain.

**In softmax, pipeline fragments within a head.** A fragment's scores and exponentials do not depend on the previous fragment's output product. Flywheel interleaves them and keeps the fragments in one compiler basic block, so the scheduler can overlap matrix and vector work. The pipeline also continues across head boundaries.

RPA-v3 interleaves work across KV heads. Under GQA with high tensor parallelism, a device may have only one KV head, leaving no head-level work to overlap. Flywheel's overlap comes from consecutive fragments of the same head, so it survives that case. With the static anchor held fixed, this pipeline raises the 32K-token ablation from **521 to 722 TFLOP/s**.

**In GDN, overlap chunk-local work with recurrent work.** The inverse depends only on a chunk's own inputs. It does not need the incoming recurrent state. While the first chunk's inverse runs, we can build the second chunk's local matrices. While the second inverse runs, we can compute the first chunk's state-dependent output and update. The recurrence stays ordered, but independent work fills the inverse's latency gaps.

<figure class="figure-compact">
  <a href="{{ '/assets/posts/2026-09-20-attention-tpu/paper/schedule.pdf' | relative_url }}"><img src="{{ '/assets/posts/2026-09-20-attention-tpu/paper/schedule.png' | relative_url }}" alt="Schedules overlapping softmax fragments and independent GDN chunk computations." width="1800" height="1358" loading="lazy"></a>
  <figcaption>Figure 3. Top: overlap each softmax fragment’s QK and exponentials with the preceding fragment’s PV. Bottom: fill the gaps between GDN inverse products with independent work from neighboring chunks. Bar lengths are schematic.</figcaption>
</figure>

<details markdown="1" id="fragment-schedule">
<summary>Fragment Pipelining</summary>

For one head, the software pipeline has the following schematic order. `weights` uses the checked static-anchor path; the range checks and guarded replay are omitted here.

```python
p = weights(scores(fragment[0]))
for i in range(1, num_fragments):  # fully unrolled
    s_next = scores(fragment[i])  # next QK product
    accumulate_output(p, fragment[i - 1].V)  # previous PV product
    p = weights(s_next)          # next exponentials
accumulate_output(p, fragment[-1].V)
```

These are operations exposed to the compiler, rather than a claim that all three execute simultaneously. Their independent matrix and vector instructions can overlap within the basic block. The final output product drains the pipeline. This independence also exists with online softmax; the fixed anchor makes each pipelined step cheaper by removing the recurring rescale.

For GDN, $\mathbf{A}\_{kk}$, $\mathbf{A}\_{qk}$, and $\mathbf{U}^{-1}$ are chunk-local. The computations of $\mathbf{V}\_{\mathrm{new}}$, $\mathbf{O}$, and the state increment use the incoming $\mathbf{S}$. Interleaving across these two groups fills gaps while preserving the order of recurrent state updates.

</details>

The changes work together. Layout gives each operand more useful work; the reformulations remove vector bottlenecks; the pipeline overlaps the dependencies that remain. With online softmax and no fragment pipeline, the ablation reaches 353 TFLOP/s. With both the checked anchor and the pipeline, it reaches 722 TFLOP/s.

## Additional Implementation Details

The main techniques set the direction. Making them work inside a model also requires handling ragged batches, preprocessing, and the TPU's tile geometry. We collect those details here so they do not interrupt the argument above.

<details markdown="1" id="layout-details">
<summary>Ragged Boundaries and Projection Layouts</summary>

A dense per-head tile is convenient for the MXU, but ragged sequences can begin at arbitrary token positions. On v6e, the DMA window follows an 8-row alignment. Flywheel snaps the start down to the aligned row. Before writing, it restores any preceding sequence's output rows from the previous buffer in on-chip vector memory (VMEM). Figure 1b shows a sequence beginning at row 13 whose window starts at row 8. Rows 8–12 are copied back before the DMA, avoiding a separate HBM gather.

For softmax, projections produce and consume the head-major layout directly. The logical input shapes are $\mathbf{Q}\in\mathbb{R}^{T\times H\_q\times D}$ and $\mathbf{K},\mathbf{V}\in\mathbb{R}^{T\times H\_{kv}\times D}$; the kernel organizes each head as a dense token-by-channel tile.

GDN-v3's wrapper requires FP32 inputs in `[T, 1, dim]` layout, adding upcasting and retiling. Flywheel consumes the projection's `[T, dim]` BF16 output and returns packed output for the next projection. Its kernel transposes $\mathbf{K}$ once and reuses it in the shared-operand products. Packing two 128-wide heads into a 256-wide matrix operand also increases the work done per load.

</details>

<details markdown="1" id="gdn-preprocessing">
<summary>Convolution and Vector Operations</summary>

GDN applies a four-tap causal convolution before the recurrence. With tokens on sublanes, shifting by one token requires vector rearrangement. Flywheel instead stacks tap-scaled inputs and multiplies by a constant selector matrix, as in Figure 2b:

$$
\mathbf{Y}=\mathbf{S}_{\mathrm{conv}}\mathbf{W}.
$$

For $\mathbf{X}\in\mathbb{R}^{C\times128}$ and $k$ taps, $\mathbf{W}\in\mathbb{R}^{kC\times128}$ contains the tap-scaled inputs and the preceding chunk's halo. The selector $\mathbf{S}\_{\mathrm{conv}}\in\{0,1\}^{C\times kC}$ picks and sums the required rows. It is labeled $\mathbf{S}$ in the figure, but is distinct from the recurrent state. For $k=4$ and $C=64$, the contraction dimension is 256. This trades extra matrix arithmetic for fewer vector shifts and accumulations.

SiLU and query/key normalization each use one specialized-unit operation (`tanh` and `rsqrt`, respectively), alongside the arithmetic and reduction shown below:

$$
\begin{aligned}
\operatorname{SiLU}(x)&=\frac{x}{2}(1+\tanh(x/2)),\\
\frac{\mathbf{x}}{\sqrt{\|\mathbf{x}\|_2^2+\epsilon}}
&=\mathbf{x}\,\operatorname{rsqrt}\!\left(\textstyle\sum_j x_j^2+\epsilon\right).
\end{aligned}
$$

The normalization uses $\epsilon=10^{-6}$. The paper measures these changes together with the inverse and convolution reformulations in the reduced-vector-work ablation, rather than attributing a standalone speedup to each one.

</details>

<details markdown="1" id="tile-details">
<summary>Exponential Evaluation and Tile Sizes</summary>

The softmax equations above use natural exponentials, matching the paper. The implementation uses `exp2`: it folds $\log\_2 e$ and the attention scale into $\mathbf{Q}$ before the fragment loop, so no per-score conversion is needed there. Anchors and score margins are converted consistently.

A 2048×2048 softmax block contains 32 fragments of 256 query rows and 512 key rows. The loop is unrolled into one compiler basic block. When a block contains multiple heads, the last output product of one head overlaps the first score product of the next.

GDN uses 128-row tiles containing two 64-token chunks when they fit. On one v7x TensorCore, the 48-value-head case only fits a single chunk per tile in its 64 MiB of VMEM. That prevents the cross-chunk pipeline, a limitation visible in the results below.

</details>

## Evaluation


We measure the Flywheel kernels at two levels: kernel-level speedup in isolation, and engine-level speedup inside vLLM’s `tpu-inference` on long-context applications, including needle-in-a-haystack retrieval, multi-hop reasoning over long documents, and video understanding.

Kernel measurements use one v6e chip or one v7x TensorCore. Their BF16 peaks are 918 and 1153 TFLOP/s, respectively. A v7x chip has two TensorCores, so these are not whole-chip comparisons across generations.

<details markdown="1" id="benchmark-setup">
<summary>Evaluation Setup</summary>

The kernel benchmarks use one v6e chip (918 TFLOP/s BF16 peak, 128 MiB VMEM) or one v7x TensorCore (1153 TFLOP/s, 64 MiB VMEM). A v7x chip contains two TensorCores. These comparisons use JAX 0.11.0 and libtpu 0.0.44, with the baseline commits specified near the start of the post.

Softmax prefill is causal and BF16, with batch size 1, head dimension $D=256$, $H\_q=32$, and $H\_{kv}=32$ (MHA) or 4 (GQA), from 1K to 128K tokens. Kernel throughput divides analytic FLOPs by device time from a profiler trace. The roofline assumes one read of each input and one output write. Full attention blocks include projections and layout work and use wall-clock time.

GDN benchmarks time the complete fused convolution and prefill call, including wrapper operations, with 16 query/key heads, $d\_k=d\_v=128$, and $n\_v\in\{16,32,48\}$ value heads. Activations are BF16 and the recurrent state is FP32. Calls contain one or eight equal-length sequences. Kernel times are medians over five rounds of back-to-back calls.

Application runs serve Qwen3.8-27B in vLLM. The v6e setup uses eight chips with eight-way tensor parallelism through 256K tokens, and sixteen chips with eight-way tensor parallelism and two-way prefill context parallelism at 512K. The v7x runs use four chips (eight TensorCores) with eight-way tensor parallelism throughout. Ratios compare the kernels within each setup.

For RULER and BABILong, prefill throughput is prompt tokens divided by generation wall time. Video-MME reports the mean language-model prefill time per question. The v7x evaluation uses 20 samples per RULER/BABILong task and eight Video-MME questions; v6e Video-MME evaluates 674 questions. At 512K tokens, and on v7x also at 256K, the paper extends RoPE with YaRN at factor 2.

</details>

### Kernel Performance

<figure>
  <a href="{{ '/assets/posts/2026-09-20-attention-tpu/paper/softmax_kernel.pdf' | relative_url }}"><img src="{{ '/assets/posts/2026-09-20-attention-tpu/paper/softmax_kernel.png' | relative_url }}" alt="Softmax attention throughput on v6e and v7x for MHA and GQA." width="1800" height="850" loading="lazy"></a>
  <figcaption>Figure 4. Softmax attention forward throughput at head dimension 256: one v6e chip (top) and one v7x TensorCore (bottom), under MHA (32 query / 32 KV heads) and GQA (32 / 4). Ratios beneath each length compare Flywheel with RPA-v3 and Splash.</figcaption>
</figure>

Figure 4 measures causal attention forward passes at head dimension 256 and batch size 1, with 32 query heads and either 32 KV heads (MHA) or 4 (GQA). It times the attention kernel alone. On v6e, Flywheel reaches 744 TFLOP/s, 81% of peak BF16 throughput. From 2K to 128K tokens, it is 3.11–3.49× faster than RPA-v3 under MHA and 1.50–1.73× under GQA. Against Splash Attention, it is 1.10–1.56× faster over the same range. On v7x, the corresponding gains are 3.57–4.14× under MHA and 1.57–1.66× under GQA over RPA-v3, and 1.18–1.80× over Splash Attention.

Our throughput barely depends on the number of KV heads: from 8K on, MHA and GQA differ by at most 1.2%, and Splash Attention behaves similarly. RPA-v3 is the opposite. It keeps heads on sublanes, so its blocks shrink as KV heads are added; under MHA, padding also leaves half its BF16 rows unused. This is where the layout choice shows up in the measurements.

The gain is not limited to these two shapes. Across 792 v6e configurations varying batch size, head count, KV head count and head dimension, Flywheel beats both baselines in 785, with a median speedup of 1.30× over the faster baseline. The seven exceptions are at 1K tokens and head dimension 256, where it stays within 9% of Splash. These are kernel-only timings; the full-block comparison below includes the projections and layout work around them.

For GDN prefill we time the whole fused convolution and recurrence, including the wrapper’s data preparation (Figure 5).

<figure>
  <a href="{{ '/assets/posts/2026-09-20-attention-tpu/paper/gdn_kernel.pdf' | relative_url }}"><img src="{{ '/assets/posts/2026-09-20-attention-tpu/paper/gdn_kernel.png' | relative_url }}" alt="Throughput of fused causal convolution and GDN prefill on v6e and v7x." width="1800" height="889" loading="lazy"></a>
  <figcaption>Figure 5. Fused conv1d + GDN prefill: 16 query/key heads, head dimension 128, and 16, 32, or 48 value heads. Tokens are packed into one or eight sequences. The ratios compare Flywheel with GDN-v3 on the same device.</figcaption>
</figure>

On v6e the speedup is 1.84–2.25× at 1K tokens, 2.45–3.10× at 4K and 2.88–3.61× from 8K on. With 48 value heads, Flywheel processes 9.9M tokens/s at 16K against 3.0M for GDN-v3. The incremental 8K-token ablation gives 1.52–1.68× from coordinated layouts, then 1.06–1.22× from operand reuse, 1.51–1.58× from reduced vector work, and 1.13–1.25× from pipelining. These are successive gains, not standalone speedups.

On one v7x TensorCore, Flywheel is 1.56–2.62× faster than GDN-v3. Its 64 MiB of VMEM also exposes a limit: at 48 value heads only one 64-token chunk fits per tile, preventing cross-chunk pipelining. More compute and bandwidth do not remove the need for enough on-chip space to keep independent work in flight.

### End-to-End Application

Kernel speedups do not always survive a serving stack, so we ran the kernels inside vLLM on a model that exercises both. Qwen3.8-27B interleaves 48 GDN layers with 16 softmax attention layers. Flywheel replaces the baseline softmax, GDN and vision-encoder attention kernels within the same build; the model, scheduler and other operators stay fixed.

<figure>
  <a href="{{ '/assets/posts/2026-09-20-attention-tpu/paper/e2e_split.pdf' | relative_url }}"><img src="{{ '/assets/posts/2026-09-20-attention-tpu/paper/e2e_split.png' | relative_url }}" alt="Qwen3.8-27B application results on RULER, BABILong and Video-MME for v6e and v7x." width="1800" height="870" loading="lazy"></a>
  <figcaption>Figure 6. Qwen3.8-27B in vLLM on v6e (top) and v7x (bottom). RULER and BABILong report prompt tokens divided by generation wall time; Video-MME reports language-model prefill time. The v7x evaluation uses 20 samples per RULER/BABILong task and eight Video-MME questions.</figcaption>
</figure>

Figure 6 shows the results on both generations. On v6e, both backends score 100% on RULER's evaluated needle-in-a-haystack tasks at every length. Prefill throughput improves by 1.07× at 4K, 1.23× at 128K and 1.53× at 512K. BABILong at 128K, 256K and 512K shows 1.22×, 1.29× and 1.56×, with identical accuracy scores for the two backends. On v7x, both text workloads reach 1.35× speedup at 512K; RULER is slightly slower at 4K (0.98×), where attention is a much smaller share of the work.

Why are the end-to-end gains smaller than the kernel gains, and why do they grow with context? Amdahl's law explains the difference. At 4K most of a prefill step goes to the projections and the MLP, and attention is a small slice of it. As the context grows, softmax attention grows quadratically and takes over the step. The Video-MME panels show the same effect on prompts with roughly 113K and 225K visual tokens.

On long videos, the v6e LLM prefill segment gets 1.24× faster at 1024 frames (4.80 s to 3.87 s) and 1.29× at 2048 frames (11.60 s to 9.00 s). On v7x the corresponding gains are 1.15× and 1.24×. These are language-model prefill times, not total video processing time.

Accuracy remains comparable on the evaluated workloads. On v6e, Video-MME scores are identical at 2048 frames and differ by one of 674 questions at 1024 frames. On the smaller v7x evaluation, both runs give identical Video-MME answers; mean RULER scores differ by at most 0.8 points, and BABILong scores match at 128K and 256K and differ by one point at 512K.

### Softmax Attention Block Time

Kernel-only timings exclude the cost of preparing inputs and consuming outputs. We also measure from hidden states to hidden states, including the projections and all layout conversions. Under MHA on v6e, Flywheel is 1.48× faster than RPA-v3 at 1K and 2.95× at 128K. Gains over Splash are smaller because its surrounding projections also avoid separate layout conversions.

<details markdown="1" id="full-block-results">
<summary>Attention Block Breakdown</summary>

Figure 4 compares the attention kernels alone, but a kernel-only timing leaves the cost of producing its input layout outside the clock. For Flywheel and Splash Attention, the surrounding projections can directly produce and consume the attention layout. RPA-v3 reads from a paged KV cache that interleaves heads, so its block also concatenates and transposes K and V. Figure 7 therefore times the whole attention block, from hidden states to hidden states with $d\_{\mathrm{model}} = 8192$, so both projections and every layout conversion are inside the clock.

<figure>
  <a href="{{ '/assets/posts/2026-09-20-attention-tpu/paper/softmax_block.pdf' | relative_url }}"><img src="{{ '/assets/posts/2026-09-20-attention-tpu/paper/softmax_block.png' | relative_url }}" alt="Full attention block time split into projections, layout work and the attention kernel." width="1800" height="542" loading="lazy"></a>
  <figcaption>Figure 7. Attention block time on one v6e chip, with 32 query heads and head dimension 256. R, S and F are RPA-v3, Splash and Flywheel. Bars are split into QKV projection, layout, kernel and output projection, and normalized to the slowest block of each group.</figcaption>
</figure>

The layout segment appears only on RPA-v3: under MHA it costs 0.25 ms at 1K and 33.7 ms at 128K. The main source of the performance gap changes with sequence length. At 1K, layout work accounts for 65% of the MHA gap; at 128K, the kernel accounts for 96%. Flywheel finishes the 128K MHA block in 466 ms against 1371 ms for RPA-v3, 2.95× faster. The speedup grows from 1.48× at 1K because attention grows quadratically with sequence length while the projections grow only linearly: attention is 3% of the block's FLOPs at 1K and 80% at 128K. Under GQA the speedup is 1.41–1.76×.

Splash also avoids layout conversions, so the full-block gains over it are smaller: 1.01–1.15× from 2K onward. At 1K, where projections account for about 90% of the block, Splash's block is 4% faster.

</details>


## Conclusion and Next Steps

The gains are largest in prefill, where there is enough matrix work to amortize operand loads and overlap neighboring fragments or chunks. Short sequences leave less room for either. The v7x GDN case also shows that on-chip capacity can limit which pipeline fits, even on a faster device.

Decode is ongoing work. With only one query token per head, reading the KV cache or recurrent state often dominates the step, so the priorities differ from prefill. We are continuing to develop and benchmark these paths in the [Flywheel codebase](https://github.com/heavyball-research/flywheel-tpu).

We have added TPU support for [ReplaySSM](https://tridao.me/blog/2026/replayssm/), which caches recent inputs alongside a recurrent-state checkpoint instead of writing the full state at every step. Flywheel applies this approach to both GDN and Kimi Delta Attention (KDA), including speculative verification. We encourage readers to read the ReplaySSM team's post for the algorithm and its motivation.

Some milestones in the codebase so far:

- **ReplaySSM for GDN and KDA:** [the initial implementation](https://github.com/heavyball-research/flywheel-tpu/pull/13) adds decode and verification support; [a subsequent update](https://github.com/heavyball-research/flywheel-tpu/pull/21) stores the FP32 checkpoint as high and low 16-bit planes, allowing ordinary decode steps to read only the high plane.
- **FP8 KV caches:** [FP8 cache support](https://github.com/heavyball-research/flywheel-tpu/pull/22) adds `float8_e4m3fn` and `float8_e5m2` storage, followed by [head-pair fragments and v7x tuning](https://github.com/heavyball-research/flywheel-tpu/pull/30).
- **Speculative verification:** [short verification windows](https://github.com/heavyball-research/flywheel-tpu/pull/23) share staged key/value blocks by folding the window into the query heads, with [v7x configurations](https://github.com/heavyball-research/flywheel-tpu/pull/33) added for these workloads.

For the current APIs, benchmark instructions, and subsequent improvements, see the [README](https://github.com/heavyball-research/flywheel-tpu#readme) and [release history](https://github.com/heavyball-research/flywheel-tpu/releases). The repository will continue to evolve as this work progresses.

## Acknowledgments

This work was supported by a 2026 Google TPU Research Award and the TPU Builders Program. The authors thank Josh Gordon, Lauren Jensen, Edgar Chen, Satha Kumar, Tasha Allen, Aditi Joshi, and the entire TPU Builders Support Team for their help with TPU onboarding.

## References

[1] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
{: #ref-1}

[2] T. Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. ICLR 2024. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
{: #ref-2}

[3] J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, T. Dao. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. NeurIPS 2024. [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
{: #ref-3}

[4] T. Zadouri, M. Hoehnerbach, J. Shah, T. Liu, V. Thakkar, T. Dao. FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling. 2026. [arXiv:2603.05451](https://arxiv.org/abs/2603.05451)
{: #ref-4}

[5] K. Hong, G. Dai, J. Xu, Q. Mao, X. Li, J. Liu, K. Chen, Y. Dong, Y. Wang. FlashDecoding++: Faster Large Language Model Inference on GPUs. MLSys 2024. [arXiv:2311.01282](https://arxiv.org/abs/2311.01282)
{: #ref-5}

[6] Z. Ye et al. FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving. MLSys 2025. [arXiv:2501.01005](https://arxiv.org/abs/2501.01005)
{: #ref-6}

[7] S. Yang, Y. Zhang. FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism. 2024. [github.com/fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
{: #ref-7}

[8] J. Jiang, Y. Chen, B. A. Hechtman, F. Zhang, Y. Mu. Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU. 2026. [arXiv:2604.15464](https://arxiv.org/abs/2604.15464)
{: #ref-8}

[9] A. Sobczyk, G. Gottardo, C. K. Matzoros, M. De Vita, F. Skogh, A. Zouzias, J. Zhuang. Fast and Stable Triangular Inversion for Delta-Rule Linear Transformers. 2026. [arXiv:2605.21325](https://arxiv.org/abs/2605.21325)
{: #ref-9}

[10] S. Yang, J. Kautz, A. Hatamizadeh. Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
{: #ref-10}

[11] S. Yang, B. Wang, Y. Zhang, Y. Shen, Y. Kim. Parallelizing Linear Transformers with the Delta Rule over Sequence Length. NeurIPS 2024. [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)
{: #ref-11}

[12] SemiAnalysis. TPU Inference Externalization Full Steam Ahead: InferenceX. September 2026. [newsletter.semianalysis.com](https://newsletter.semianalysis.com/p/tpu-inferencex-full-steam)
{: #ref-12}

[13] NVIDIA. CUDA Programming Guide 13.1, Compute Capabilities, Table 30. [docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/compute-capabilities.html)
{: #ref-13}

[14] NVIDIA. NVIDIA H100 Tensor Core GPU Architecture, whitepaper, Tables 1 and 4. 2022. [resources.nvidia.com](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c)
{: #ref-14}

[15] NVIDIA. CUDA C++ Programming Guide 12.8, Compute Capability 9.0 and Hardware Multithreading. [docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#compute-capability-9-0)
{: #ref-15}

[16] NVIDIA. H100 Tensor Core GPU specifications. [nvidia.com](https://www.nvidia.com/en-us/data-center/h100/)
{: #ref-16}

[17] NVIDIA. DGX B200 datasheet. [resources.nvidia.com](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)
{: #ref-17}

[18] NVIDIA. CUDA Programming Guide 13.1, Asynchronous Data Copies (TMA). [docs.nvidia.com](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html)
{: #ref-18}

[19] N. P. Jouppi, D. H. Yoon, G. Kurian, S. Li, N. Patil, J. Laudon, C. Young, D. Patterson. A Domain-Specific Supercomputer for Training Deep Neural Networks. Communications of the ACM 63(7), 2020. [doi:10.1145/3360307](https://doi.org/10.1145/3360307)
{: #ref-19}

[20] N. P. Jouppi, S. Lakshmanamurthy, C. Young, D. Patterson. Google's Training Supercomputers from TPU v2 to Ironwood: Architectural Stability, Scale, Resilience, Power Efficiency, and Sustainability Across Five Generations. IEEE Micro, July/August 2026. [arXiv:2606.15870](https://arxiv.org/abs/2606.15870)
{: #ref-20}

[21] OpenXLA. Tiled layout. [openxla.org](https://openxla.org/xla/tiled_layout)
{: #ref-21}

[22] JAX. Writing TPU kernels with Pallas. [docs.jax.dev](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
{: #ref-22}

[23] Google Cloud. TPU architecture, and the TPU v4, v5e, v5p, v6e and TPU7x pages. [docs.cloud.google.com](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
{: #ref-23}

[24] J. Austin, S. Douglas, R. Frostig, A. Levskaya, C. Chen, S. Vikram, F. Lebron, P. Choy, V. Ramasesh, A. Webson, R. Pope. How to Think About TPUs. In How To Scale Your Model. 2025. [jax-ml.github.io/scaling-book](https://jax-ml.github.io/scaling-book/tpus/)
{: #ref-24}

[25] P. Toulme. From JAX to VLIW: Tracing a Computation Through the TPU Compiler Stack. December 2025. [patricktoulme.substack.com](https://patricktoulme.substack.com/p/from-jax-to-vliw-tracing-a-computation)
{: #ref-25}

[26] P. Toulme. When XLA Isn't Enough, with public LLO compiler dumps for TPU v6e. January 2026. [patricktoulme.substack.com](https://patricktoulme.substack.com/p/when-xla-isnt-enough-from-pallas), [github.com/patrick-toulme/justabyte](https://github.com/patrick-toulme/justabyte/tree/main/tpu_pallas_post)
{: #ref-26}

[27] JAX. Mosaic TPU vector layout passes, `infer_vector_layout.cc` and `apply_vector_layout.cc`, jax-v0.8.0. [github.com/jax-ml/jax](https://github.com/jax-ml/jax/tree/jax-v0.8.0/jaxlib/mosaic/dialect/tpu/transforms)
{: #ref-27}

[28] NVIDIA. Parallel Thread Execution ISA: warpgroup-level matrix multiply-accumulate (`wgmma`) and TensorCore 5th generation (`tcgen05`) instructions. [docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/)
{: #ref-28}

[29] OpenXLA. Tokamax: Splash Attention for TPU. [github.com/openxla/tokamax](https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/experimental/tpu/splash_attention).
{: #ref-29}

[30] vLLM Project. `gdn_v3`, the Gated DeltaNet kernel in tpu-inference. [github.com/vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference)
{: #ref-30}
