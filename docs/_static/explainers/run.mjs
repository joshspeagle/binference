// Brings the explainers on a page to life. Each <figure data-explainer="v1"> arrives with its
// still frame and disabled controls (docs/_ext/bnf_explainer.py); this enables the controls and
// redraws the frame from the module of the same name on every change.
for (const fig of document.querySelectorAll("figure[data-explainer]")) {
  import(`./${fig.dataset.explainer}.mjs`).then((ex) => {
    const plot = fig.querySelector(".explainer-plot"), out = fig.querySelector(".explainer-readout");
    const inputs = [...fig.querySelectorAll("input[name]")];
    const update = () => {
      const state = Object.fromEntries(inputs.map((i) => [i.name, Number(i.value)]));
      inputs.forEach((i) => {
        const c = ex.controls.find((c) => c.name === i.name);
        i.nextElementSibling.textContent = Number(i.value).toFixed(c.digits);
      });
      plot.innerHTML = ex.draw(state);
      out.textContent = ex.readout(state);
    };
    inputs.forEach((i) => { i.disabled = false; i.addEventListener("input", update); });
  });
}
