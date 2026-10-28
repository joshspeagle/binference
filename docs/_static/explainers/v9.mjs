// V-9: Polya-tree draws by depth. Each level halves every bin with a Beta split whose
// concentration c h^2 grows with the depth h, so finer levels fluctuate less.
import { fmt, linspace, plot, polya, rng, svg } from "./core.mjs";

const N = 1000, SHOWN = [["brand", "draw 1"], ["cat1", "draw 2"], ["cat2", "draw 3"]];

export const title = "Pólya-tree draws by depth";
export const controls = [
  { name: "levels", label: "levels", min: 1, max: 5, step: 1, value: 4 },
  { name: "logc", label: "log₁₀ c", min: -1, max: 1, step: 0.05, value: 0 },
];

// N draws of the 2^levels bin fractions of PolyaTree(c, power=2) on [0, 1], flat base.
export function compute({ levels, logc }) {
  const c = 10 ** logc, u = rng(9);
  return { c, edges: linspace(0, 1, 2 ** levels + 1), draws: Array.from({ length: N }, () => polya(levels, c, u)) };
}

// The standard deviation of a bin's density over the draws, and exactly: a bin's fraction is
// the product of one share of each split, so E rho^2 = prod_h (a_h + 1) / (2 (2 a_h + 1)),
// with a_h = c h^2, and E rho = 2^-levels.
export function readout({ levels }, { c, draws }) {
  const K = 2 ** levels, x = draws.flat().map((r) => K * r); // densities, pooled over bins
  const sd = Math.sqrt(x.reduce((a, v) => a + (v - 1) ** 2, 0) / x.length);
  let m2 = 1;
  for (let h = 1; h <= levels; h++) m2 *= (c * h * h + 1) / (2 * (2 * c * h * h + 1));
  return `c = ${+c.toPrecision(2)}, ${K} bins: SD of a bin's density ${fmt(sd, 2)} (exact ${fmt(K * Math.sqrt(m2 - 1 / K ** 2), 2)})`;
}

export function draw({ levels }, { edges, draws }) {
  const ymax = Math.max(1, ...draws.slice(0, SHOWN.length).flat().map((r) => r * (edges.length - 1)));
  const p = plot({ edges, ymax, ylabel: "density", key: [["ink dashed", "prior mean"], ...SHOWN] });
  const alt = `Three draws of a Pólya tree with ${levels} ${levels === 1 ? "level" : "levels"}, ` +
    `${edges.length - 1} bins, as step densities around the flat prior mean.`;
  return svg(p.h, alt, SHOWN.map(([cls], i) => p.masses(draws[i], cls)).join("") +
    p.line([0, 1], [1, 1], "ink dashed") + p.axes());
}
