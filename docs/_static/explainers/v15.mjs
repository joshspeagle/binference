// The three-bin tutorial's trial population, with two observation mechanisms.
import { fmt, plot, svg } from "./core.mjs";

const EDGES = [0, 1, 2, 4];
const RHO = [0.25, 0.5, 0.25];
const EFFICIENCY = [0.8, 0.5, 0.25];
const ROW_A = [0.4, 0.2, 0.04];  // the first galaxy's likelihood row on a probability scale
const dot = (x, y) => x.reduce((sum, value, k) => sum + value * y[k], 0);

export const title = "Selection and response on three bins";
export const controls = [
  { name: "mode", label: "Selection acts", options: ["On the truth", "On the measurement"], value: 0 },
];

export function compute({ mode }) {
  const rowA = ROW_A.map((value, k) => value * (mode === 0 ? EFFICIENCY[k] : 1));
  const rowB = EFFICIENCY.map((value, k) => value - rowA[k]);
  const missed = EFFICIENCY.map((value) => 1 - value);
  const normaliser = dot(EFFICIENCY, RHO);
  const selected = EFFICIENCY.map((value, k) => value * RHO[k] / normaliser);
  return {
    edges: EDGES, rho: RHO, efficiency: EFFICIENCY, rawRow: ROW_A,
    response: [rowA, rowB], missed, selected, normaliser,
    probabilityA: dot(rowA, RHO) / normaliser,
  };
}

export function readout({ mode }, { normaliser, probabilityA }) {
  const entry = mode === 0 ? "On the truth" : "On the measurement";
  return `${entry}: observed share ${fmt(normaliser, 4)}, P(A | observed) ${fmt(probabilityA, 3)}.`;
}

export function draw(state, c) {
  const p = plot({
    edges: EDGES, h: 190, ymax: 0.6, ylabel: "density",
    key: [["brand thin", "population ρ"], ["sample dashed", "observed φ"]],
  });
  const values = [
    ["Unadjusted A row", c.rawRow],
    ["Recorded in A", c.response[0]],
    ["Recorded in B", c.response[1]],
    ["Missed", c.missed],
  ];
  const text = (x, y, value, align = "start") =>
    `<text class="key" x="${x}" y="${y}" text-anchor="${align}">${value}</text>`;
  const columns = [204, 276, 352];
  const tableTop = p.h + 24;
  const header = ["0–1", "1–2", "2–4"].map((name, i) =>
    text(columns[i], tableTop, name, "middle")
  ).join("");
  const rows = values.map(([name, row], n) => {
    const y = tableTop + 25 * (n + 1);
    return text(14, y, name) + row.map((value, k) =>
      text(columns[k], y, fmt(value, 2), "middle")
    ).join("");
  }).join("");
  const alt = `${readout(state, c)} The solid population and dashed observed densities ` +
    `sit above a table of conditional probabilities with true bins as columns. ` +
    `Rows in order are unadjusted A, recorded A, recorded B and missed. ` +
    values.map(([, row]) => row.map((value) => fmt(value, 2)).join(", ")).join("; ");
  return svg(tableTop + 114, alt, p.masses(c.rho, "brand thin") +
    p.masses(c.selected, "sample dashed") + p.axes() + header + rows);
}
