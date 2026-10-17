---
layout: page
---

HeavyBall Research is a research group spanning [NYU Shanghai](https://shanghai.nyu.edu/) and [NYU Tandon](https://engineering.nyu.edu/), led by [Yucheng Lu](https://www.yucheng-lu.me/index.html).

Our mission is to build the next generation of AI systems. We work broadly on efficient modeling, hardware-aware training and inference algorithms, and the systems that bring them together at scale.

We warmly welcome passionate researchers who are excited about the intersection of systems and AI to [join us](https://www.yucheng-lu.me/joinus.html)!

## Group Updates

- [2026-10] Thanks Amazon for supporting our research through the **2026 Amazon Research Award**!
- [2026-09] [Multi-token Residual Prediction](./blog/2026-07-01-multi-token-residual-prediction) is accepted by NeurIPS'26, congrats to the team!
- [2026-04] Thanks Google for supporting our research through the **2026 Google TPU Research Award**!
- [2026-04] We are delighted to welcome three new PhD students (Zishuo, Yufeng, and Runyu) to our group!

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
