// V-1: one object's likelihood over z and its bin averages, for a chosen error and bin width.
import { binAverages, fmt, plot, svg } from "./core.mjs";

const D = 1.37, Z = [0, 3];

export const title = "Bin averages of one likelihood";
export const controls = [
  { name: "sigma", label: "error σ", min: 0.05, max: 0.6, step: 0.05, value: 0.2, digits: 2 },
  { name: "width", label: "bin width", min: 0.1, max: 1, step: 0.05, value: 0.25, digits: 2 },
];

// The bin averages L_k of a Gaussian likelihood centred on D, on bins of the given width
// from 0 (the last bin is cut at 3), at the likelihood's own scale.
export function compute({ sigma, width }) {
  const K = Math.ceil((Z[1] - Z[0]) / width - 1e-9);
  const edges = Array.from({ length: K + 1 }, (_, k) => Math.min(Z[0] + k * width, Z[1]));
  const L = binAverages(D, sigma, edges);
  return { edges, L, share: L.filter((l, k) => l * (edges[k + 1] - edges[k]) > 0.05).length };
}

export function readout(state) {
  const { share } = compute(state);
  return `${share} ${share === 1 ? "bin holds" : "bins hold"} more than 5% of the likelihood`;
}

export function draw(state) {
  const { sigma } = state, { edges, L } = compute(state);
  const peak = 1 / (sigma * Math.sqrt(2 * Math.PI));
  const p = plot({ h: 250, x: Z, y: [0, 8.2], xlabel: "true value z", ylabel: "likelihood",
    xticks: [0, 1, 2, 3], yticks: [0, 4, 8] });
  const zs = Array.from({ length: 241 }, (_, i) => (Z[1] * i) / 240);
  const ys = zs.map((z) => Math.min(peak * Math.exp(-0.5 * ((z - D) / sigma) ** 2), 8.2));
  const alt = `A dashed bell curve of width ${fmt(sigma, 2)} centred on z = ${D}, and steps at its ` +
    `average over each of ${L.length} bins. ${readout(state)}.`;
  return svg(p.h, alt,
    edges.map((e) => p.vline(e, "grid")).join("") +
    p.stairs(edges, L, "fill-sample") + p.stairs(edges, L, "sample") +
    p.line(zs, ys, "sample dashed") + p.axes() +
    p.key(240, 26, [["sample dashed", "likelihood ℓ(z)"], ["sample", "bin averages L"]]));
}
