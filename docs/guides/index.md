# User guide

The user guide explains how binference is organised and how to make good choices.
The [tutorials](../tutorials/index) show complete analyses, and the
[theory](../how-it-works/theory) page has the maths.

An analysis has four steps, and each has a page:

1. **Describe your data** as per-object likelihoods over bins: {doc}`inputs`.
2. **Choose a prior** for the population ({doc}`priors`) and, if your
   catalogue is incomplete, a selection function ({doc}`selection`).
3. **Run an engine**: {doc}`engines`.
4. **Use the result**: population summaries, derived quantities and per-object
   posteriors ({doc}`results`). Then check the result ({doc}`validation`).

{doc}`performance` covers large problems, and {doc}`pitfalls` lists the
mistakes that silently bias results.

```{toctree}
:maxdepth: 1

inputs
priors
selection
engines
results
validation
performance
pitfalls
```
