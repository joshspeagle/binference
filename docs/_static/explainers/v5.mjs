// V-5: shrinkage. One object's posterior is its likelihood times the population, so a narrow
// population pulls it in and a broad one leaves it close to the likelihood.
import { binAverages, density, normalise, fmt, linspace, meanZ, plot, svg } from "./core.mjs";

const EDGES = linspace(0, 3, 31), D = 1.9, SIGMA = 0.4, CENTRE = 1.2;
const L = binAverages(D, SIGMA, EDGES);
const mean = (m) => meanZ(m, EDGES);

export const title = "Shrinkage toward the population";
export const controls = [
  { name: "width", label: "population width", min: 0.1, max: 1.5, step: 0.05, value: 0.35, digits: 2 },
];

// The population's masses (a normal at 1.2 binned and cut to [0, 3]) and the object's
// posterior masses r_k ∝ L_k rho_k (PopulationModel.responsibilities).
export function compute({ width }) {
  const rho = normalise(binAverages(CENTRE, width, EDGES));
  return { edges: EDGES, L, rho, posterior: normalise(L.map((l, k) => l * rho[k])) };
}

export function readout(state) {
  const { posterior } = compute(state);
  return `measured at ${D}; posterior mean ${fmt(mean(posterior), 2)}`;
}

export function draw(state) {
  const { rho, posterior } = compute(state), d = (m) => density(m, EDGES);
  const p = plot({ h: 260, x: [0, 3], y: [0, 2.4], xlabel: "true value z", ylabel: "density",
    xticks: [0, 1, 2, 3], yticks: [0, 1, 2] });
  const alt = `A population of width ${fmt(state.width, 2)} centred on ${CENTRE}, one object's ` +
    `likelihood centred on ${D}, and its posterior between them. ${readout(state)}.`;
  return svg(p.h, alt,
    p.stairs(EDGES, d(rho).map((x) => Math.min(x, 2.4)), "fill-brand") +
    p.stairs(EDGES, d(rho).map((x) => Math.min(x, 2.4)), "brand thin") +
    p.stairs(EDGES, d(normalise(L)), "sample dashed") + p.stairs(EDGES, d(posterior), "sample") + p.axes() +
    p.key(230, 24, [["brand thin", "population ρ"], ["sample dashed", "likelihood"], ["sample", "posterior"]]));
}
