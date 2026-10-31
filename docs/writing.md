# Writing the docs

These rules hold for every user page and docstring. The pages under `docs/design/`
record how binference was designed and are exempt.

## Readers

Write for these readers, in this order:

1. students and non-experts with data in hand;
2. practitioners in any field;
3. expert statisticians.

Astronomy leads the examples. Every input type and every field still has its own way in.

## Sentences

- Keep sentences to at most 25 words and instructions to at most 20.
- Put one idea in each sentence.
- Use the active voice.
- Put a condition before its instruction: "If the chains disagree, run them longer."

## Words and symbols

- Use one word for each meaning, as defined in the [glossary](reference/glossary.md).
  A field's own names appear only on that field's pages, as synonyms.
- Use one symbol for each meaning, as defined on the [notation page](reference/notation.md).

## Facts

- Give each fact one home. Other pages link to it instead of repeating it.
- Defaults and constants live in the Reference section, not in the theory pages.
- State only measured figures, each with the setup that produced it. Every number on a
  page comes from code that CI runs, or from a benchmark run outside CI. The page names
  that run's script, options and date.
- Do not cite design records ("DECISIONS #n") in user pages or docstrings. The reason
  for a design belongs in its issue or pull request. Docstrings do not cite them; a
  citation belongs in a comment beside the code.

## Concept pages

- Open with the question the page answers.
- Ask the reader to run no code; code belongs in tutorials and guides.
- Add at most one "In plain words" note, with the `{plain-words}` directive:

  ````md
  ```{plain-words}
  Each object's estimate borrows strength from the whole sample.
  ```
  ````

## Figures

Figures, explainers and diagrams follow the [drawing rules](design/brand.md#drawing-rules)
of the design page.
