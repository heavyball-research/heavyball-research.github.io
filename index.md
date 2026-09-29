---
layout: page
---

HeavyBall Research is a research group in [Computer Science at NYU Shanghai](https://shanghai.nyu.edu/), led by [Yucheng Lu](https://www.yucheng-lu.me/index.html).

Our mission is to advance the frontier of AI and machine learning systems. We work on a range of projects such as:

- Efficient training and inference algorithms for foundation models.
- Hardware/software co-design for efficient modeling.
- High-performance GPU kernels for modern AI workloads.
- Large-scale AI application systems (e.g., Agent frameworks, RAG systems).

We collaborate closely with industry partners and other research labs to build cutting-edge AI systems, and are well-supported with abundant compute resources and research funding.

We warmly welcome passionate researchers who are excited about the intersection of systems and AI to [join us](https://www.yucheng-lu.me/joinus.html)!

## Group Updates

- [April 2026] Thanks Google for supporting our research through the **2026 Google TPU Research Award**!
- [April 2026] We are delighted to welcome three new PhD students (Zishuo, Yufeng, and Runyu) to our group!

## Latest posts

<ul class="post-list compact">
{% for post in site.posts limit:5 %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <span class="meta">{{ post.date | date: "%b %-d, %Y" }}</span>
  </li>
{% endfor %}
</ul>

[All posts →]({{ '/blog/' | relative_url }})
