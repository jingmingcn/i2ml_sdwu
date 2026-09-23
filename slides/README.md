# Introduction to Machine Learning — Course Slides

Slide decks for the course **Introduction to Machine Learning**, based on
*Introduction to Machine Learning with Python* by Andreas C. Müller & Sarah Guido
(O'Reilly, 2016).

Slides are written in **[Marp](https://marp.app/)** — plain Markdown that exports
to PDF, PPTX, or HTML. Edit any `.md` file in a text editor; no proprietary format.

## Decks (one per chapter)

| File | Content |
|------|---------|
| `00-course-overview.md` | Course intro & roadmap |
| `01-introduction.md` | Ch.1 — ML concepts, tools, first model |
| `02-supervised-learning.md` | Ch.2 — Classification & regression algorithms |
| `03-unsupervised-learning.md` | Ch.3 — Preprocessing, dim. reduction, clustering |
| `04-data-representation.md` | Ch.4 — Feature engineering & selection |
| `05-model-evaluation.md` | Ch.5 — Cross-validation, grid search, metrics |
| `06-pipelines.md` | Ch.6 — Pipelines & avoiding data leakage |
| `07-text-data.md` | Ch.7 — Bag-of-words, tf-idf, topic modeling |
| `08-wrapping-up.md` | Ch.8 — Workflow, production, next steps |

Shared styling lives in `themes/course.css` — edit once, every deck updates.

## How to edit

- Each `---` on its own line starts a **new slide**.
- The block at the top of each file (between `---` fences) is front-matter/config.
- Change colors, fonts, or sizes in `themes/course.css`.
- Add images to `images/` and reference them: `![w:600](images/pic.png)`.

## How to build / preview

### Option A — VS Code (easiest)
Install the **"Marp for VS Code"** extension. Open any `.md` file and click the
preview icon. Use the export button to save PDF/PPTX/HTML.

### Option B — Command line
Requires [Node.js](https://nodejs.org/). No install needed thanks to `npx`:

```bash
# from the slides/ directory

# Live preview in the browser (auto-reloads on save)
npx @marp-team/marp-cli --theme themes/course.css -p -w 01-introduction.md

# Export ONE deck to PDF
npx @marp-team/marp-cli --theme themes/course.css 01-introduction.md --pdf

# Export to PowerPoint
npx @marp-team/marp-cli --theme themes/course.css 01-introduction.md --pptx

# Export ALL decks to PDF at once
for f in [0-9][0-9]-*.md; do
  npx @marp-team/marp-cli --theme themes/course.css "$f" --pdf
done
```

PDFs/PPTX are written next to each source file.

> Tip: install once globally with `npm i -g @marp-team/marp-cli` to drop the `npx`.
