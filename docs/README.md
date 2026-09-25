# GitHub Pages

The published front page is generated from [`original/strategy-guide.md`](../original/strategy-guide.md).
Edit the canonical guide, then run from the repository root:

```sh
python3 scripts/build_pages.py
python3 scripts/build_pages.py --check
```

The script preserves the guide's content, gives section headings stable IDs,
and resolves repository-relative evidence links to GitHub. It validates local
link targets and requires no third-party Python packages. `docs/index.md` is
generated; do not edit it separately.

GitHub Pages uses the `main` branch, `/docs` folder, and its native Jekyll build.
The repository URL and project-site base path are in `_config.yml`. The page
layout, stylesheet, and optional navigation/table enhancement script are local
assets with no external font, analytics, or JavaScript dependencies. Content and
anchor navigation remain available without JavaScript.
