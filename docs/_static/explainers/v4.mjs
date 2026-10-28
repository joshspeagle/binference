// V-4: stacking is one EM step. Step through EM from a flat population; step 1 is the stack.
import { density, emStep, fmt, linspace, mock, plot, svg, truth } from "./core.mjs";

const EDGES = linspace(0, 3, 21), MAX = 200;
const { L } = mock(800, 0.3, EDGES, 11);
const TRUTH = truth(EDGES), PATH = [EDGES.slice(1).map(() => 1 / 20)]; // PATH[t] is step t
for (let t = 1; t <= MAX; t++) PATH.push(emStep(L, PATH[t - 1]));
const loglike = (rho) => L.reduce((s, row) => s + Math.log(row.reduce((a, l, k) => a + l * rho[k], 0)), 0) / L.length;

export const title = "Stacking, then EM";
export const controls = [{ name: "step", label: "EM step", min: 1, max: MAX, step: 1, value: 1, digits: 0 }];

export function compute({ step }) {
  return { edges: EDGES, L, flat: PATH[0], rho: PATH[step], loglike: loglike(PATH[step]) };
}

export function readout({ step }) {
  const here = `mean log-likelihood per object ${fmt(loglike(PATH[step]), 3)}`;
  return step === 1 ? `step 1 is the stack: ${here}` : `step ${step}: ${here} (the stack: ${fmt(loglike(PATH[1]), 3)})`;
}

export function draw(state) {
  const { rho } = compute(state), d = (m) => density(m, EDGES);
  const p = plot({ h: 270, x: [0, 3], y: [0, 1.6], xlabel: "true value z", ylabel: "population density",
    xticks: [0, 1, 2, 3], yticks: [0, 0.5, 1, 1.5] });
  const alt = `EM step ${state.step} for 800 mock objects. The stack, step 1, is broad and misses ` +
    `the dip between the two peaks of the dashed truth; later steps sharpen the peaks. ${readout(state)}.`;
  return svg(p.h, alt,
    p.stairs(EDGES, d(PATH[1]), "seq") + p.stairs(EDGES, d(TRUTH), "ink dashed") +
    p.stairs(EDGES, d(rho), "brand") + p.axes() +
    p.key(220, 24, [["seq", "step 1, the stack"], ["ink dashed", "truth"], ["brand", `step ${state.step}`]]));
}
