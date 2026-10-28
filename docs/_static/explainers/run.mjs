// Brings the explainers on a page to life. Each <figure data-explainer="v1"> arrives with its
// still frame and disabled controls (docs/_ext/bnf_explainer.py); once it comes within 300 px
// of the viewport, this imports the module of the same name, enables the controls and redraws
// the frame from it, once per animation frame while they move. Nothing moves on its own, so
// reduced motion needs no special case. Screen readers hear the readout once, when a control
// is released, not once per frame.
function start(fig) {
  import(`./${fig.dataset.explainer}.mjs`).then((ex) => {
    const plot = fig.querySelector(".explainer-plot"), out = fig.querySelector(".explainer-readout");
    const live = fig.querySelector(".explainer-live");
    const inputs = [...fig.querySelectorAll("input[name], select[name]")];
    const sliders = inputs.filter((i) => i.type === "range").map((i) => [i, fig.querySelector(`output[for="${i.id}"]`)]);
    const state = () => Object.fromEntries(inputs.map((i) => [i.name, Number(i.value)]));
    let pending = false;
    const update = () => {
      pending = false;
      for (const [i, o] of sliders) {
        o.textContent = Number(i.value).toFixed((i.step.split(".")[1] ?? "").length);
        i.setAttribute("aria-valuetext", o.textContent);
      }
      const s = state(), c = ex.compute(s);
      plot.innerHTML = ex.draw(s, c);
      out.textContent = ex.readout(s, c);
    };
    update(); // a value the browser restored on reload is drawn at once
    for (const i of inputs) {
      i.disabled = false;
      i.addEventListener("input", () => { if (!pending) { pending = true; requestAnimationFrame(update); } });
      // the readout of the last computed result: this frame runs after the input event's
      i.addEventListener("change", () => requestAnimationFrame(() => { live.textContent = out.textContent; }));
    }
  });
}

const near = new IntersectionObserver((entries) => {
  for (const e of entries.filter((e) => e.isIntersecting)) { near.unobserve(e.target); start(e.target); }
}, { rootMargin: "300px" });
for (const fig of document.querySelectorAll("figure[data-explainer]")) near.observe(fig);
