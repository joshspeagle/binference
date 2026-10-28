// V-3: posteriors used as likelihoods. Each object's posterior is its likelihood times an
// interim prior; a fit that skips dividing the prior out is pulled toward it.
import { binAverages, emPath, fmt, linspace, meanZ, mock, normalise, plot, posterior, svg, truth } from "./core.mjs";

const EDGES = linspace(0, 3, 16), STEPS = 50, PRIOR_WIDTH = 0.4;
const { L } = mock(2000, 0.3, EDGES, 3);
const fit = (rows) => emPath(rows, STEPS)[STEPS];
const RIGHT = fit(L), TRUTH = truth(EDGES), mean = (m) => fmt(meanZ(m, EDGES), 2);
const cache = new Map(); // one entry per slider position: each costs 50 EM steps

export const title = "Posteriors used as likelihoods";
export const controls = [
  { name: "centre", label: "interim prior peak", min: 0.3, max: 2.7, step: 0.1, value: 2.4 },
];

// The interim prior's masses (a binned normal at the slider's centre), the objects'
// posteriors under it, and the EM fits with the prior divided out and without.
export function compute({ centre }) {
  if (!cache.has(centre)) {
    const prior = normalise(binAverages(centre, PRIOR_WIDTH, EDGES).map((p) => Math.max(p, 1e-12)));
    const posteriors = L.map((row) => posterior(row, prior));
    cache.set(centre, { L, prior, posteriors, right: RIGHT, wrong: fit(posteriors), steps: STEPS });
  }
  return cache.get(centre);
}

export const readout = (state, { wrong }) =>
  `mean z: truth ${mean(TRUTH)}, prior divided out ${mean(RIGHT)}, posteriors as likelihoods ${mean(wrong)}`;

export function draw({ centre }, { prior, wrong }) {
  const ymax = Math.max(1.2, ...wrong.map((m) => m / (EDGES[1] - EDGES[0])));
  const p = plot({ edges: EDGES, ymax, ylabel: "density", key: [["swatch-ink", "interim prior"],
    ["ink dashed", "truth"], ["brand", "prior divided out"], ["cat1", "used as likelihoods"]] });
  const alt = `Population fits of ${L.length} mock objects. With the prior divided out the fit ` +
    `follows the dashed truth. With posteriors used as likelihoods it is pulled toward the ` +
    `interim prior at ${fmt(centre, 1)}.`;
  return svg(p.h, alt, p.masses(prior, "fill-ink") + p.masses(TRUTH, "ink dashed") +
    p.masses(RIGHT, "brand") + p.masses(wrong, "cat1") + p.axes());
}
