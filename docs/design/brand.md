# The binference design

This page is the locked design of the binference documentation site. Josh locked it on
2026-10-27 (issue #99) from the round-2 design sheet; the rounds that led to it are kept in
`notes/review/docs-design/`. Its numbers live in one place,
[`docs/_scripts/brand.py`](https://github.com/joshspeagle/binference/blob/main/docs/_scripts/brand.py),
which draws every mark and writes the site's colour tokens; this page explains them.
A change to the design changes `brand.py` first, then this page.

## Concept

binference infers a population and gives every answer with its uncertainty. The design
draws that idea: a population is shown as its posterior draws, never as one histogram;
objects and samples are single coral points; bins are square steps.

## The mark

The mark is a letter b built as a histogram. The tallest bin is the stem. The other bins
trace a bowl: a low floor plus a semi-ellipse that spans them exactly. Seven posterior
draws of the histogram are overlaid as pale fills, so the b is darkest where they agree
and its edge steps where they disagree. One line is their mean. An elliptical counter is
cut out of the bowl and holds one sample, in coral.

The mark has three cuts, each drawn from the same geometry:

| Cut | Used at | Bins | Draws | File |
|---|---|---|---|---|
| full | 48 px and up | 10 | 7 | `_static/brand/mark-full-{light,dark}.svg` |
| mid | 24 to 47 px | 7 | 4 | `_static/brand/mark-mid-{light,dark}.svg` |
| tiny | below 24 px, favicons | 5 | 2 | `_static/brand/mark-tiny-{light,dark}.svg`, `favicon.svg` |

The favicon follows the browser's colour scheme. Rules for the mark:

- Draws are always the brand colour and the sample always coral; never swap them.
- Never fill the b solid, outline it only, rotate it or redraw it in other hues.
- Clear space around the mark is at least a quarter of its height.

## The wordmark

"binference" is set in Sora at weight 600, lower case. The dot of the i is a measurement
with an error bar, in coral. The lockups in `_static/brand/lockup-{light,dark}.svg` hold
the wordmark as outlines, so they never depend on a loaded font. The mark is 1.45 times
the wordmark's font size, and both share a baseline.

## Colour

One brand hue, indigo, is the population: draws, bins, estimates and the mark. Coral is
an object or a sample, and nothing else. Neutrals lean toward indigo. Both themes are
designed; the dark theme is not an inversion.

The contrast column is the WCAG ratio: for surfaces (`bg`, `surface`, `raised`, `bar`)
against the text on them, for `bar-text` against the bar, and for the rest against the page. Text needs 4.5 (AA), marks and large text 3.
`sample` (3.2 in light) is for marks only; coral text uses `sample-text`.

<!-- tokens:start -->
| Token | Role | Light | Contrast | Dark | Contrast |
|---|---|---|---|---|---|
| `bg` | page | `#FBFBFE` | 15.4 | `#10112A` | 15.4 |
| `surface` | panels, code, explainers | `#F1F2FA` | 14.2 | `#181A3D` | 13.9 |
| `raised` | hover, selected | `#E6E8F5` | 13.0 | `#22255A` | 11.8 |
| `rule` | hairlines (decorative) | `#DADCEC` | 1.3 | `#2B2E5E` | 1.5 |
| `ink` | text | `#1B1D4A` | 15.4 | `#E8E9FB` | 15.4 |
| `ink-2` | secondary text, axes | `#5D5F86` | 5.9 | `#9C9ECB` | 7.2 |
| `brand` | the population: draws, bins, the mark | `#4A4FC4` | 6.3 | `#8C91F0` | 6.5 |
| `link` | links, focus | `#3A3FB0` | 8.1 | `#A5A9F5` | 8.4 |
| `sample` | an object or a sample (marks only) | `#EC5F3E` | 3.2 | `#FF8A6B` | 8.0 |
| `sample-text` | an object or a sample, as text | `#C2410C` | 5.0 | `#FF8A6B` | 8.0 |
| `bar` | top bar and footer, both themes | `#10112A` | 15.4 | `#0B0C20` | 16.1 |
| `bar-text` | text on the bar | `#E8E9FB` | 15.4 | `#E8E9FB` | 16.1 |
| `tip` | 'In plain words' notes | `#0E7D6E` | 4.9 | `#4CC2B0` | 8.5 |
| `warn` | warnings | `#9A6200` | 4.9 | `#E8AE52` | 9.3 |
| `danger` | errors | `#B42318` | 6.4 | `#F97066` | 6.6 |
| `cat-0…4` | categories, in order | `#4A4FC4` `#C07C00` `#1A9E8C` `#8F2E74` `#5E8A1C` | | `#7C82E8` `#C4871C` `#25A390` `#9C4A8E` `#749E39` | |
| `seq-0…4` | sequential | `#E2E3F8` `#B8BBEE` `#8186DE` `#4C53C2` `#2A2F8F` | | `#1F2257` `#33388C` `#5259C7` `#8A90EC` `#C9CCFA` | |
| `div-0…4` | diverging | `#2A2F8F` `#8186DE` `#E4E4EA` `#E0A848` `#8A5600` | | `#C9CCFA` `#5C63C9` `#3A3B4E` `#C48A2A` `#F2C77A` | |
<!-- tokens:end -->

The data colours passed the categorical checks of the dataviz palette validator in both
themes (lightness band, chroma floor, colour-blind separation of adjacent pairs, normal-vision
separation, contrast with the page): categories in this order, never cycled; a ninth series
folds into "other" or a separate panel. Sequential ramps use one hue; diverging ramps have a
grey middle.

## Type

| Role | Face | Size |
|---|---|---|
| Wordmark, page titles, headings | Sora 600 | titles 32 px, sections 20 px |
| Text | Atkinson Hyperlegible Next 400 | 16.5 px, line height 1.6 |
| Code, numbers in tables | Atkinson Hyperlegible Mono 400 | 14 px |
| Captions | Atkinson Hyperlegible Next 400 | 14 px |

Atkinson Hyperlegible was designed by the Braille Institute so that letters that look alike
(I l 1, O 0, rn m) stay distinct. All three faces are under the SIL Open Font License and are
served from `_static/fonts/` (fetched by `docs/_scripts/fetch_fonts.py`). They have no Greek
letters, so Greek symbols are always written as maths (`$\rho$`), which MathJax sets in its
own fonts.

## Drawing rules

These rules hold for every figure, explainer and diagram on the site.

1. **Uncertainty is drawn, never hidden.** A population is shown as its posterior draws
   (six to eight pale fills, or a band) with the mean as one line. A bare histogram is never
   an answer.
2. **Indigo is the population, coral is an object.** Draws, bins and estimates are indigo;
   data points, samples and single objects are coral; the truth is an ink dashed line.
   Coral marks nothing else.
3. **Bins are steps.** Anything binned is drawn as square steps, never smoothed. Smooth
   curves are only for what is continuous, such as a likelihood or a true density.
4. **Square and flat.** Corners are square (at most 2 px on controls). No gradients,
   shadows or glows. Hairline rules, not boxes.
5. **Words before colour.** Direct labels over legends. Hue is never the only signal: use
   fill and outline, dot and line, dashed and solid. Every figure reads in greyscale.
6. **Readable on a phone.** Figure text is at least 12 px at 390 px wide. A wide figure has
   a narrow version. Every figure has a caption and real alt text.

## Components

- **Top bar** in `bar` (dark indigo) in both themes, with the lockup, the seven sections,
  search with the `/` key, the theme toggle and a GitHub link. On phones it holds the
  lockup, search and a menu.
- **Sidebar** with the section's pages; the current page has a 3 px brand rule.
- **"In plain words"** note: one per Concepts page at most, in `tip`, for students.
- **Notes and warnings**: a 3 px rule in `brand` or `warn` on `surface`.
- **Code** on `surface`, with a language label and a copy button; **output** joined below
  it on `bg`, labelled "output".
- **Figures**: a hairline above the caption; captions in `ink-2`.
- **Explainers**: on `surface` with a 3 px `brand` rule on top, a numbered title (V-n),
  native controls, and a live readout in mono. They are still and complete before
  JavaScript runs, and respect reduced motion.
- **Pager** with the previous and next pages; **footer** in `bar`.
- **Home**: the hero is the full mark with its draws slowly resampling (still under reduced
  motion), the line "Infer the population, then every object.", and three paths by the
  reader's data.

## Accessibility

WCAG 2.2 AA in both themes: the contrast above, a visible 2 px focus ring in `link`, a skip
link, one `h1` per page, alt text on every figure, and nothing that moves without a way to
stop it. The final pass checks every page at 390 and 1440 px in both themes.
