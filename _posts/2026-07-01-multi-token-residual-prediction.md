---
title: "Multi-token Residual Prediction"
authors:
  - name: Yufeng Xu
    url: https://twitter.com/Zephyr271828
  - name: Rahul Chalamala
    url: https://twitter.com/rchalamala
  - name: Yucheng Lu
    url: https://twitter.com/_yucheng_lu
tldr: "A tiny module that predicts the residual between denoising steps of a diffusion LM: up to 1.56× lossless speedup in static decoding and up to +16 accuracy points recovered in dynamic decoding."
external_url: https://modal.com/blog/multi-token-residual-prediction
image: /assets/posts/2026-07-01-multi-token-residual-prediction/headline-results.png
links:
  - label: Paper
    url: https://arxiv.org/abs/2605.18817
  - label: Code
    url: https://github.com/heavyball-research/multi-token-residual-prediction
  - label: SGLang
    url: https://github.com/heavyball-research/sglang
  - label: Models
    url: https://huggingface.co/collections/heavyball/sdar-mrp
  - label: Original post on Modal
    url: https://modal.com/blog/multi-token-residual-prediction
---

> Editor's Note: this guest blog post describes the results of a research collaboration between Modal Research and NYU Shanghai's [HeavyBall Research](https://heavyball-research.github.io/). It first appeared on the [Modal blog](https://modal.com/blog/multi-token-residual-prediction).

<figure>
  <img src="{{ '/assets/posts/2026-07-01-multi-token-residual-prediction/headline-results.png' | relative_url }}" alt="MRP headline results">
  <figcaption>MRP is one small module with two uses. Left: in the static regime, it accelerates decoding losslessly (speculative, lookahead steps K = 3) or further still with a small quality cost (direct, lookahead steps K = 1), reaching up to 1.56× throughput in SGLang. Right: in the dynamic regime, it recovers up to +16 accuracy points lost to aggressive low-threshold decoding (threshold τ = 0.5). Averaged over GSM8K, MATH500, HumanEval, and MBPP on SDAR-1.7B/4B/8B.</figcaption>
</figure>

## TL;DR

Multi-Token Prediction (MTP) speeds up autoregressive models by predicting several tokens from a single forward pass. We adapt this to diffusion language models by training a module to predict the residual between denoising steps rather than full distributions. This tiny module enables both lossless speedup during static decoding (up to 1.56×) and quality recovery in dynamic decoding (up to +16 accuracy points).

**Paper**: <https://arxiv.org/abs/2605.18817>

**Code**: <https://github.com/heavyball-research/multi-token-residual-prediction>

**SGLang implementation**: <https://github.com/heavyball-research/sglang>

**Models**: <https://huggingface.co/collections/heavyball/sdar-mrp>

## Starting from MTP

Autoregressive models generate one token per forward pass, and that's expensive. Lightweight prediction heads like [Medusa](https://arxiv.org/abs/2401.10774), [EAGLE](https://arxiv.org/abs/2401.15077), and [DeepSeek's MTP](https://arxiv.org/abs/2412.19437) predict multiple tokens from hidden states, combined with speculative verification for real speedups.

We asked whether this approach could work beyond autoregressive models. Diffusion LMs start from masked sequences and gradually unmask high-confidence positions — building in parallelism but creating a tradeoff: unmask too many tokens per step and quality degrades since tokens lack context from others being revealed simultaneously. Most acceleration research operates on this single quality-speed curve.

## A naïve attempt

The direct approach: train a small head to predict the next step's full log-density from current hidden states, then apply it multiple times. Results collapsed quickly:

| Method    | K=1  | K=2  | K=3 | K=4 |
| --------- | ---- | ---- | --- | --- |
| Naïve MTP | 84.8 | 16.9 | 5.9 | 1.9 |

*GSM8K (0-shot, CoT) accuracy on SDAR-4B. Per-step errors compounded, destroying accuracy beyond step one.*

## The main insight

If the full next-step distribution is hard to predict across multiple steps, the question is whether there is an easier target to predict instead. The backbone's output changes only slightly between adjacent denoising steps. Rather than predicting complete distributions from scratch, predict small corrections to already-good predictions.

This stems from the Markov structure of denoising: each step touches only a few positions, so by a Lipschitz argument the predictions at untouched positions can only shift minimally. The signal a small module must learn is inherently low-complexity.

## Multi-token Residual Prediction (MRP)

<figure>
  <img src="{{ '/assets/posts/2026-07-01-multi-token-residual-prediction/arch-diagram.png' | relative_url }}" alt="MRP architecture diagram">
</figure>

A small 3-layer transformer reads the frozen backbone's hidden states, predicts inter-step logit residuals, and adds them to the backbone's logits. Only the MRP module trains; the backbone, language head, and embeddings remain frozen.

The training objective uses KL divergence on masked positions, comparing the backbone's output before and after revealing tokens. Because softmax normalizing constants cancel, this matches true conditional distributions while the module only represents the correction.

| Method    | K=1  | K=2  | K=3  | K=4  |
| --------- | ---- | ---- | ---- | ---- |
| Naïve MTP | 84.8 | 16.9 | 5.9  | 1.9  |
| MRP       | 88.6 | 84.9 | 70.9 | 57.2 |

*GSM8K (0-shot, CoT) accuracy on SDAR-4B. The residual framing dramatically improves multi-step prediction. At K = 2, residual learning leads by 65+ points.*

## Applications in inference

DLMs typically run in two regimes: **static denoising** unmasks fixed numbers of positions per step (high quality, low throughput), and **dynamic denoising** unmasks all positions exceeding a confidence threshold (high throughput, lower quality). MRP serves both, playing different roles in each.

### Application I: Lossless speedup in static denoising

MRP creates an inference dial with multiple operating points. At one extreme is **speculative decoding**: MRP cheaply proposes tokens, the backbone verifies them in one pass, and positions where they agree are accepted. Positions where the backbone agrees with the draft are accepted; positions where it disagrees are remasked and redone. Verification hidden states seed the next iteration, amortizing costs when acceptance rates are high.

| Backbone | GSM8K        | MATH500      | HumanEval    | MBPP         |
| -------- | ------------ | ------------ | ------------ | ------------ |
| SDAR-4B  | 90.0 / 1.36x | 68.0 / 1.26x | 67.7 / 1.35x | 66.5 / 1.27x |
| SDAR-8B  | 90.4 / 1.40x | 74.8 / 1.39x | 72.6 / 1.34x | 67.3 / 1.34x |

*Speculative mode in SGLang. Accuracy (%) followed by throughput speedup. Quality matches the backbone by construction.*

At the other extreme is **direct decoding**: skipping verification and committing MRP's corrected logits directly. This raises the speedup ceiling, and residual predictions prove accurate enough on reasoning tasks that the quality cost is small:

| Setting             | GSM8K        | MATH500      | HumanEval    | MBPP         |
| ------------------- | ------------ | ------------ | ------------ | ------------ |
| Baseline            | 90.9 / 1x    | 72.2 / 1x    | 73.8 / 1x    | 67.7 / 1x    |
| Direct (MRP Step 1) | 90.1 / 1.59x | 71.4 / 1.61x | 67.1 / 1.53x | 63.8 / 1.51x |
| Direct (MRP Step 2) | 89.2 / 1.89x | 70.8 / 1.91x | 64.0 / 1.78x | 59.9 / 1.75x |

*Direct decoding on SDAR-8B. Accuracy (%) followed by throughput speedup. Tokens commit without verification. K=1 stays within a point on reasoning at 1.6×; K=2 exceeds 1.8× with larger code-task drops.*

Users choose the operating point based on task requirements. This control only exists when you manage the inference stack directly, not behind closed APIs where the provider fixes the policy.

### Application II: Quality recovery

In aggressive low-threshold dynamic decoding, the backbone over-commits many tokens per step. Each token is chosen before seeing others revealed simultaneously, causing quality loss.

Here MRP reverses direction: after backbone over-reveals, a single MRP pass predicts residuals conditioned on fresh reveals. Any token whose confidence now falls below the threshold is remasked and deferred to a later step with more context. MRP reads how predictions shift once new neighbors are accounted for, identifying over-confident isolated reveals. The same threshold τ gates both the reveal and the remask, so no additional tuning is introduced.

Prior work like [DMax](https://arxiv.org/abs/2604.08302), [RCD](https://arxiv.org/abs/2601.22954), and [WINO](https://arxiv.org/abs/2507.18578) revisit reveals and remask low-confidence tokens, but spend full backbone passes re-evaluating. MRP performs the same check via a single lightweight residual pass.

| Model | τ   | GSM8K               | MATH500             | HumanEval           | MBPP                |
| ----- | --- | ------------------- | ------------------- | ------------------- | ------------------- |
| 1.7B  | 0.5 | 41.6 → 59.1 (+17.5) | 26.0 → 37.4 (+11.4) | 17.7 → 28.7 (+11.0) | 26.9 → 41.3 (+14.4) |
| 1.7B  | 0.6 | 56.3 → 67.0 (+10.7) | 33.4 → 40.4 (+7.0)  | 31.7 → 43.3 (+11.6) | 42.4 → 49.0 (+6.6)  |
| 1.7B  | 0.7 | 65.4 → 71.8 (+6.4)  | 39.4 → 48.6 (+9.2)  | 40.9 → 45.1 (+4.2)  | 49.8 → 51.0 (+1.2)  |
| 1.7B  | 0.8 | 70.6 → 75.4 (+4.8)  | 47.4 → 52.0 (+4.6)  | 45.7 → 48.8 (+3.1)  | 51.8 → 51.8 (0.0)   |
| 1.7B  | 0.9 | 76.2 → 77.3 (+1.1)  | 51.2 → 57.0 (+5.8)  | 49.4 → 52.4 (+3.0)  | 53.7 → 54.1 (+0.4)  |
| 4B    | 0.5 | 63.4 → 81.1 (+17.7) | 44.2 → 58.4 (+14.2) | 32.3 → 53.1 (+20.8) | 38.5 → 50.6 (+12.1) |
| 4B    | 0.6 | 76.4 → 85.5 (+9.1)  | 53.6 → 61.4 (+7.8)  | 49.4 → 57.9 (+8.5)  | 49.8 → 57.2 (+7.4)  |
| 4B    | 0.7 | 84.6 → 88.5 (+3.9)  | 60.4 → 65.6 (+5.2)  | 60.4 → 62.2 (+1.8)  | 61.1 → 63.4 (+2.3)  |
| 4B    | 0.8 | 87.9 → 90.1 (+2.2)  | 66.8 → 70.6 (+3.8)  | 64.6 → 62.8 (−1.8)  | 63.8 → 64.2 (+0.4)  |
| 4B    | 0.9 | 88.5 → 90.1 (+1.6)  | 69.0 → 70.6 (+1.6)  | 67.1 → 65.9 (−1.2)  | 65.4 → 64.6 (−0.8)  |
| 8B    | 0.5 | 67.9 → 82.3 (+14.4) | 45.2 → 58.0 (+12.8) | 32.3 → 54.9 (+22.6) | 34.6 → 49.4 (+14.8) |
| 8B    | 0.6 | 79.6 → 86.8 (+7.2)  | 54.8 → 63.8 (+9.0)  | 48.8 → 63.4 (+14.6) | 48.3 → 59.9 (+11.6) |
| 8B    | 0.7 | 85.9 → 89.0 (+3.1)  | 60.8 → 69.0 (+8.2)  | 64.6 → 72.6 (+8.0)  | 54.9 → 60.3 (+5.4)  |
| 8B    | 0.8 | 89.3 → 91.0 (+1.7)  | 68.0 → 70.2 (+2.2)  | 74.4 → 75.0 (+0.6)  | 62.3 → 66.5 (+4.2)  |
| 8B    | 0.9 | 90.8 → 91.4 (+0.6)  | 70.0 → 72.0 (+2.0)  | 75.0 → 75.0 (0.0)   | 66.9 → 68.5 (+1.6)  |

*MRP remasking improves accuracy of low-threshold dynamic decoding. Gains peak at aggressive thresholds (up to +22.6 on 8B HumanEval at τ = 0.5) and shrink as τ rises and the backbone already unmasks conservatively. Reasoning improves at every operating point.*

## What we learned

Several insights emerged:

- **Depth has a sweet spot at 2–3 layers.** Accuracy improves steadily from 1 to 3 layers, then plateaus while throughput keeps dropping. By 8 layers the tradeoff is strictly worse. Other speculative modules like [DFlash](https://arxiv.org/abs/2602.06036) go deeper (around 5 layers) using per-layer KV injection feeding hidden states into every draft layer, enabling greater benefit from additional depth.

- **The best depth depends on the mode.** In direct decoding, where MRP outputs are used directly, deeper modules help more. In speculative decoding, verification corrects MRP's mistakes anyway, so a shallower and faster module reaches the same final quality at higher throughput.

- **MRP composes cleanly.** The module provides a primitive that other methods can build on. At its core, MRP is just a cheap, accurate prediction of what the next denoising step would produce. The same module supports both speculative decoding and remasking, adding capabilities rather than competing with existing approaches.

For diffusion LM work, MRP is easy to attach and provides previously unavailable choices: speed without quality loss, or even greater speed while recovering otherwise-lost quality.
