// V-3: posteriors used as likelihoods. Each object's posterior is its likelihood times an
// interim prior; a fit that skips dividing the prior out is pulled toward it.
import { binAverages, emPath, fmt, linspace, meanZ, mock, normalise, plot, posterior, svg, truth } from "./core.mjs";

const EDGES = linspace(0, 3, 16), STEPS = 50, BIN = EDGES[1] - EDGES[0];
const { L } = mock(2000, 0.3, EDGES, 3);
const fit = (rows) => emPath(rows, STEPS)[STEPS];
const RIGHT = fit(L), TRUTH = truth(EDGES), mean = (m) => fmt(meanZ(m, EDGES), 2);
const fits = new Map(); // the fit at each slider position: each costs 50 EM steps

export const title = "Posteriors used as likelihoods";
export const controls = [
  { name: "centre", label: "interim prior peak", min: 0.3, max: 2.7, step: 0.1, value: 2.4 },
  { name: "width", label: "interim prior width", min: 0.2, max: 1.2, step: 0.1, value: 0.4 },
];

// The interim prior's masses (a binned normal), the objects' posteriors under it, and the EM
// fits with the prior divided out and without.
export function compute({ centre, width }) {
  const prior = normalise(binAverages(centre, width, EDGES).map((p) => Math.max(p, 1e-12)));
  const posteriors = L.map((row) => posterior(row, prior)), key = `${centre} ${width}`;
  if (!fits.has(key)) fits.set(key, fit(posteriors));
  return { L, prior, posteriors, right: RIGHT, wrong: fits.get(key), steps: STEPS };
}

export const readout = (state, { wrong }) =>
  `mean z: truth ${mean(TRUTH)}, prior divided out ${mean(RIGHT)}, posteriors as likelihoods ${mean(wrong)}`;

export function draw({ centre, width }, { prior, wrong }) {
  const ymax = Math.max(...[prior, TRUTH, RIGHT, wrong].flat()) / BIN;
  const p = plot({ edges: EDGES, ymax, ylabel: "density", key: [["swatch-ink", "interim prior"],
    ["ink dashed", "truth"], ["brand", "prior divided out"], ["cat1", "used as likelihoods"]] });
  const alt = `Population fits of ${L.length} mock objects. With the prior divided out the fit ` +
    `follows the dashed truth. With posteriors used as likelihoods it is pulled toward the ` +
    `interim prior, centred at ${fmt(centre, 1)} with width ${fmt(width, 1)}.`;
  return svg(p.h, alt, p.masses(prior, "fill-ink") + p.masses(TRUTH, "ink dashed") +
    p.masses(RIGHT, "brand") + p.masses(wrong, "cat1") + p.axes());
}
