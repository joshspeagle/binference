// V-5: shrinkage. One object's posterior is its likelihood times the population, so a narrow
// population pulls it in and a broad one leaves it close to the likelihood.
import { binAverages, fmt, linspace, meanZ, normalise, plot, posterior, svg } from "./core.mjs";

const EDGES = linspace(0, 3, 31), D = 1.9, SIGMA = 0.4, CENTRE = 1.2, WIDTH = EDGES[1] - EDGES[0];
const L = binAverages(D, SIGMA, EDGES);

export const title = "Shrinkage toward the population";
export const controls = [
  { name: "width", label: "population width", min: 0.1, max: 1.5, step: 0.05, value: 0.35 },
];

// The population's masses (a normal at CENTRE binned and cut to [0, 3]) and the object's
// posterior masses.
export function compute({ width }) {
  const rho = normalise(binAverages(CENTRE, width, EDGES));
  return { d: D, sigma: SIGMA, centre: CENTRE, edges: EDGES, L, rho, posterior: posterior(L, rho) };
}

export const readout = (state, c) => `measured at ${D}; posterior mean ${fmt(meanZ(c.posterior, EDGES), 2)}`;

export function draw({ width }, { rho, posterior: post }) {
  const p = plot({ edges: EDGES, ymax: Math.max(...rho, ...post) / WIDTH, ylabel: "density",
    key: [["brand thin", "population ρ"], ["sample dashed", "likelihood"], ["sample", "posterior"]] });
  const alt = `A population of width ${fmt(width, 2)} centred on ${CENTRE}, one object's ` +
    `likelihood centred on ${D}, and its posterior between them.`;
  return svg(p.h, alt, p.masses(rho, "fill-brand") + p.masses(rho, "brand thin") +
    p.masses(normalise(L), "sample dashed") + p.masses(post, "sample") + p.axes());
}
