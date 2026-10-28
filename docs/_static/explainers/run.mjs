// Brings the explainers on a page to life. Each <figure data-explainer="v1"> arrives with its
// still frame and disabled controls (docs/_ext/bnf_explainer.py); this enables the controls and
// redraws the frame from the module of the same name, once per animation frame while they move.
// Nothing moves on its own, so reduced motion needs no special case.
for (const fig of document.querySelectorAll("figure[data-explainer]")) {
  import(`./${fig.dataset.explainer}.mjs`).then((ex) => {
    const plot = fig.querySelector(".explainer-plot"), out = fig.querySelector(".explainer-readout");
    const inputs = [...fig.querySelectorAll("input[name]")];
    let pending = false;
    const update = () => {
      pending = false;
      const state = Object.fromEntries(inputs.map((i) => [i.name, Number(i.value)]));
      for (const i of inputs) {
        const text = Number(i.value).toFixed((i.step.split(".")[1] ?? "").length);
        fig.querySelector(`output[for="${i.id}"]`).textContent = text;
        i.setAttribute("aria-valuetext", text);
      }
      const c = ex.compute(state);
      plot.innerHTML = ex.draw(state, c);
      out.textContent = ex.readout(state, c);
    };
    for (const i of inputs) {
      i.disabled = false;
      i.addEventListener("input", () => { if (!pending) { pending = true; requestAnimationFrame(update); } });
    }
  });
}
