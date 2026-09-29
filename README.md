# HeavyBall Research — Lab Blog

Source for the lab website / blog. Built with Jekyll, hosted on GitHub Pages.

## Writing a post

1. Create `_posts/YYYY-MM-DD-short-slug.md`, using an existing post as a reference.
   The filename date is the publish date and becomes part of the URL:
   `/blog/YYYY-MM-DD-short-slug/`.
2. Fill in the front matter (`title`, `authors`, optional `tldr`, `image`, `links`)
   and set `published: true` when the post is ready to publish.
3. Put images in `assets/posts/YYYY-MM-DD-short-slug/` and reference them as
   `{{ '/assets/posts/YYYY-MM-DD-short-slug/fig.png' | relative_url }}`.
4. Markdown, LaTeX (`$...$`, `$$...$$`, `\begin{equation}`), fenced code, and raw
   HTML all work.
5. Open a PR against `main`. Once merged, GitHub Pages rebuilds in ~1–2 min.

Whole-post HTML: name the file `.html` instead of `.md`, keep the front matter.

## Local preview

Requires Ruby ≥ 3.1 (`brew install ruby` on macOS — avoid the system Ruby).

```bash
bundle install
bundle exec jekyll serve --livereload
# → http://127.0.0.1:4000
```

For work in progress, create `_drafts/` and add posts without date prefixes.
Add `--drafts` to preview them locally; normal builds omit drafts.

## Layout

```
_config.yml          site settings, permalink style, math engine
_layouts/            default.html (shell), post.html, page.html
_includes/           authors.html (author list), mathjax.html
_posts/              published posts
blog/index.html      post list
people/index.md      lab members
index.md             landing page
assets/style.css     all styling
assets/posts/<slug>/ per-post images
CNAME                custom domain (created by GitHub when you set it; don't delete)
```

## Custom domain

`_config.yml` → `url:` must match the live domain so RSS/SEO links are right.
DNS and HTTPS are configured in the repo's Settings → Pages.
