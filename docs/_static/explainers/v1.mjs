// V-1: one object's likelihood over z and its bin averages, for a chosen error and bin width.
import { binAverages, fmt, linspace, plot, svg } from "./core.mjs";

const D = 1.37, Z = linspace(0, 3, 121);

export const title = "Bin averages of one likelihood";
export const controls = [
  { name: "sigma", label: "error σ", min: 0.05, max: 0.6, step: 0.05, value: 0.2 },
  { name: "width", label: "bin width", min: 0.1, max: 1, step: 0.05, value: 0.25 },
];

// The bin averages L_k of a Gaussian likelihood centred on d, on bins of the given width
// from 0 (the last bin is cut at 3), at the likelihood's own scale.
export function compute({ sigma, width }) {
  const K = Math.ceil(3 / width - 1e-9);
  const edges = Array.from({ length: K + 1 }, (_, k) => Math.min(k * width, 3));
  const L = binAverages(D, sigma, edges);
  return { d: D, edges, L, share: L.filter((l, k) => l * (edges[k + 1] - edges[k]) > 0.05).length };
}

export const readout = (state, { share }) =>
  `${share} ${share === 1 ? "bin holds" : "bins hold"} more than 5% of the likelihood`;

export function draw({ sigma }, { edges, L }) {
  const peak = 1 / (sigma * Math.sqrt(2 * Math.PI));
  const p = plot({ edges, ymax: 1.05 * peak, ylabel: "likelihood",
    key: [["sample dashed", "likelihood ℓ(z)"], ["sample", "bin averages L"]] });
  const alt = `A dashed bell curve of width ${fmt(sigma, 2)} centred on z = ${D}, and steps at its ` +
    `average over each of ${L.length} bins.`;
  return svg(p.h, alt,
    edges.map((e) => p.vline(e, "grid")).join("") + p.stairs(L, "fill-sample") + p.stairs(L, "sample") +
    p.line(Z, Z.map((z) => peak * Math.exp(-0.5 * ((z - D) / sigma) ** 2)), "sample dashed") + p.axes());
}
