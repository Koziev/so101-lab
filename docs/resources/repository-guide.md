# Repository guide

The repository separates the website source from engineering artifacts.

| Path | Purpose |
|---|---|
| `docs/` | Website pages, blog posts, and images used in documentation |
| `hardware/` | CAD exports, printable files, wiring diagrams |
| `software/` | Reusable scripts and utilities |
| `robots/` | URDF and other robot-description files |
| `.github/workflows/` | Automated website deployment and validation |

## Adding a blog post

Create a Markdown file under `docs/blog/posts/`. Add YAML front matter with at least a date:

```yaml
---
date: 2026-11-01
categories:
  - Experiments
tags:
  - SO-101
---

# Post title
```

Use an ISO date in the filename for convenient sorting. Add images under `docs/assets/images/` and embed them using relative paths.

## Large files

Avoid committing large raw videos, datasets, and frequent sensor recordings directly to Git. Consider Git LFS or a dedicated data store for those assets. Keep generated outputs and machine-specific files out of version control.

## Updating the site

Edit Markdown locally and preview with `mkdocs serve`. Push changes to `main`; GitHub Actions builds and deploys the site.
