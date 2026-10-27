# Guides

The guides say how to do one task each. The [tutorials](../tutorials/index) show whole
analyses, and the [theory](../how-it-works/theory) page has the maths.

An analysis has five steps, and each has a guide:

1. **Prepare your inputs**: turn your data into per-object likelihoods over bins
   ({doc}`inputs/index`), and choose the bins ({doc}`bins`).
2. **Choose a prior** for the population ({doc}`priors`). If your catalogue is
   incomplete, describe its selection ({doc}`selection`).
3. **Run an engine** ({doc}`engines`). The default chooses for you.
4. **Read the result**: population summaries, derived quantities and per-object
   posteriors ({doc}`results`).
5. **Check the result**: diagnostics, then simulations ({doc}`checking`).

{doc}`performance` covers large problems, and {doc}`troubleshooting` maps each warning
and symptom to its fix.

```{toctree}
:maxdepth: 1

inputs/index
bins
priors
selection
engines
results
checking
performance
troubleshooting
```
