// V-3: posteriors used as likelihoods. Each object's posterior is its likelihood times an
// interim prior; a fit that skips dividing the prior out is pulled toward it.
import { binAverages, density, normalise, emStep, fmt, linspace, meanZ, mock, plot, svg, truth } from "./core.mjs";


const EDGES = linspace(0, 3, 16), STEPS = 300, WIDTH = 0.4;
const { L } = mock(600, 0.3, EDGES, 3);
const flat = EDGES.slice(1).map(() => 1 / 15);
const fit = (rows) => { let r = flat; for (let t = 0; t < STEPS; t++) r = emStep(rows, r); return r; };
const RIGHT = fit(L), TRUTH = truth(EDGES);
const mean = (m) => meanZ(m, EDGES);

export const title = "Posteriors used as likelihoods";
export const controls = [
  { name: "centre", label: "interim prior peak", min: 0.3, max: 2.7, step: 0.1, value: 2.4, digits: 1 },
];

// The interim prior's masses (a binned normal at the slider's centre), the objects'
// posteriors under it, and the EM fits with the prior divided out and without.
export function compute({ centre }) {
  const prior = normalise(binAverages(centre, WIDTH, EDGES).map((p) => Math.max(p, 1e-12)));
  const post = L.map((row) => normalise(row.map((l, k) => l * prior[k])));
  return { edges: EDGES, L, prior, posteriors: post, right: RIGHT, wrong: fit(post), steps: STEPS };
}

export function readout(state) {
  const { wrong } = compute(state);
  return `mean z: truth ${fmt(mean(TRUTH), 2)}, prior divided out ${fmt(mean(RIGHT), 2)}, ` +
    `posteriors as likelihoods ${fmt(mean(wrong), 2)}`;
}

export function draw(state) {
  const { prior, wrong } = compute(state), d = (m) => density(m, EDGES);
  const p = plot({ h: 270, x: [0, 3], y: [0, 1.6], xlabel: "true value z", ylabel: "density",
    xticks: [0, 1, 2, 3], yticks: [0, 0.5, 1, 1.5] });
  const alt = `Population fits of 600 mock objects. With the prior divided out the fit follows the ` +
    `dashed truth. With posteriors used as likelihoods it is pulled toward the interim prior ` +
    `at ${fmt(state.centre, 1)}. ${readout(state)}.`;
  return svg(p.h, alt,
    p.stairs(EDGES, d(prior).map((x) => Math.min(x, 1.6)), "fill-ink") +
    p.stairs(EDGES, d(TRUTH), "ink dashed") + p.stairs(EDGES, d(RIGHT), "brand") +
    p.stairs(EDGES, d(wrong), "cat1") + p.axes() +
    p.key(170, 24, [["swatch-ink", "interim prior"], ["ink dashed", "truth"],
      ["brand", "prior divided out"], ["cat1", "posteriors as likelihoods"]]));
}
