// V-4: stacking is one EM step. Step through EM from a flat population; step 1 is the stack.
import { emPath, fmt, linspace, mock, plot, svg, truth } from "./core.mjs";

const EDGES = linspace(0, 3, 21), MAX = 200, WIDTH = EDGES[1] - EDGES[0];
const { L } = mock(800, 0.3, EDGES, 11);
const TRUTH = truth(EDGES), PATH = emPath(L, MAX); // PATH[t] is step t
const YMAX = Math.max(...PATH.flat()) / WIDTH; // one scale for every step
const LOGLIKE = PATH.map((rho) => fmt(L.reduce((s, row) =>
  s + Math.log(row.reduce((a, l, k) => a + l * rho[k], 0)), 0) / L.length, 3));

export const title = "Stacking, then EM";
export const controls = [{ name: "step", label: "EM step", min: 1, max: MAX, step: 1, value: 1 }];

export const compute = ({ step }) => ({ L, rho: PATH[step] });

export function readout({ step }) {
  const here = `mean log-likelihood per object ${LOGLIKE[step]}`;
  return step === 1 ? `step 1 is the stack: ${here}` : `step ${step}: ${here} (the stack: ${LOGLIKE[1]})`;
}

export function draw({ step }, { rho }) {
  const p = plot({ edges: EDGES, ymax: YMAX, ylabel: "density",
    key: [["pale", "step 1, the stack"], ["ink dashed", "truth"], ["brand", `step ${step}`]] });
  const alt = `EM step ${step} for ${L.length} mock objects. The stack, step 1, is broad and misses ` +
    `the dip between the two peaks of the dashed truth; later steps sharpen the peaks.`;
  return svg(p.h, alt, p.masses(PATH[1], "pale") + p.masses(TRUTH, "ink dashed") +
    p.masses(rho, "brand") + p.axes());
}
