# Package name brainstorm (step 1.4, D5) — 2026-10-25

Criteria (Josh): general across fields, says something about what it does,
catchy, easy to say and remember, ideally a pun in the dynesty / emcee /
blackjax style. `pastry` (PyPI: `pastry-inference`) stays the fallback.
Availability: PyPI JSON API checked 2026-10-25 (404 = free). GitHub search was
not reachable from the session; trademarks not checked.

## Concept space

- The act: many messy per-object measurements → one population with honest
  uncertainty; hierarchical shrinkage back onto each object; stacking done right.
- Objects: bins, histograms, simplex, proportions, counts, trees, priors, urns.
- Pictures: Galton board / quincunx / bean machine (balls through noisy pegs into
  bins: our forward model; pastry runs it backwards); Pólya urn (the
  Dirichlet-multinomial); pointillism (many dots, one picture); sediment/strata;
  grain into bins (glean, winnow, harvest); polls and censuses; pastry layers.
- Hidden-acronym words: ∝ (propto), D-M (deem, taken), "pop" words
  (populace, popover, poplin), "bin" words (combine, binnacle).

## Shortlist

| Name | PyPI | Story | Risks |
|---|---|---|---|
| quincunx | free | Galton's own name for his board: pegs in a quincunx pattern scatter balls (objects) into bins (noise); the package runs the board backwards. Logo: staggered pegs. `import quincunx as qx` | spelling/pronunciation (KWIN-kunks); Galton's eugenics legacy sits behind the story (the name is the pattern's, not his) |
| propto | free | ∝: infers proportions; everything in it holds up to proportionality (likelihood rows, Bayes' rule); LaTeX users know it. emcee-style nerd pun | doesn't say "population"; search noise from LaTeX |
| popover | free | a pastry (keeps the original theme) + "pop" + "over": a posterior over the population | web-UI "popover" widget collides in search |
| populace | free | plain English for the population as a whole (vs its members, the per-object posteriors) | bland; no hint of noise or inference |
| polyurn | free | Pólya urn = the Dirichlet-multinomial in one picture; "poly" = many urns/objects; Pólya trees inside | reads oddly; funerary urns |

Honourable mentions: pointillist (free; long), unstack (free; collides with
pandas `.unstack()`, defined by negation), shoal (free; weak meaning),
millefeuille (free; unspellable).

Rejected: galton (eugenics), plinko/plinkr (trademark), beanmachine (Meta's
Bean Machine PPL), pluribus, deem, glean (taken / Mozilla Glean), historian,
pollster, census, tally, sieve, winnow, strata, mosaic, quilt (taken).

## Round 2 (2026-10-25)

Josh: no to quincunx, propto, polyurn; popover could work; populace (or a twist)
could work; wants more in the poprocks / popover vein, more plain-but-twisted
names like populace, and "inference" twists such as binference.

| Name | PyPI | Family | Story / risks |
|---|---|---|---|
| poprocks | free | pop candy | crackling crystals; pop(ulation) + "rocks". Pop Rocks is a registered candy trademark |
| cakepop | free | pastry | cake pops are crumbs pressed back into one whole: messy pieces → one population; keeps the pastry lineage |
| popover | free | pastry | round 1: pastry + posterior "over" the population; web-UI collision |
| populi | free | twist on populace | "vox populi", of the people; short (POP-yoo-lee); note vox-pop is taken |
| popscope | free | plain + twist | an instrument (-scope) for seeing a population through noisy measurements |
| popstats | free | plain | population statistics; generic |
| binference | free | inference twist | bin + inference; says exactly what it does; tied to bins (fine: bins are the interface, #17) |
| binsight | free | bin twist | bin + insight; upbeat, a little marketing-ish |
| popinfer / popinf | free | inference twist | plain; bland |
| popsterior | free | portmanteau | population posterior; memorable, maybe too jokey |
