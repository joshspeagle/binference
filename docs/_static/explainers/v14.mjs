// V-14: missing information and Gibbs mixing. The broader the likelihoods compared with the
// bins, the larger the missing information f, and the slower the Gibbs sampler mixes.
import { dirichlet, fmt, linspace, mean, mock, plot, rng, sub, svg, truth } from "./core.mjs";

const K = 10, N = 500, EDGES = linspace(0, 3, K + 1), BIN = EDGES[1] - EDGES[0], ITER = 400;
const FLAT = 0.5, RHO = truth(EDGES, FLAT); // the mock's population: every bin is occupied

// I_m = sum_i diag(r_i) - r_i r_i^T at rho, with every bin occupied; each posterior r_i is
// computed as posterior() does, into one buffer.
function infoMissing(L, rho) {
  const Im = rho.map(() => new Array(K).fill(0)), r = new Float64Array(K);
  for (const row of L) {
    let s = 0;
    for (let k = 0; k < K; k++) { r[k] = row[k] * rho[k]; s += r[k]; }
    for (let a = 0; a < K; a++) {
      Im[a][a] += r[a] / s;
      for (let b = 0; b < K; b++) Im[a][b] -= (r[a] * r[b]) / (s * s);
    }
  }
  return Im;
}

// The eigenvalues of a symmetric matrix, by cyclic Jacobi rotations.
function eigenvalues(A) {
  const n = A.length, a = A.map((r) => r.slice());
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] ** 2;
    if (off < 1e-24) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (a[p][q] === 0) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { // rotate columns p and q, then rows
        const x = a[k][p], y = a[k][q];
        a[k][p] = c * x - s * y; a[k][q] = s * x + c * y;
      }
      for (let k = 0; k < n; k++) {
        const x = a[p][k], y = a[q][k];
        a[p][k] = c * x - s * y; a[q][k] = s * x + c * y;
      }
    }
  }
  return a.map((r, i) => r[i]);
}

// The missing information f = 1 - min eig(I_obs, I_c) at rho (PopulationModel.missing_information).
// In the log-fractions, I_c = N (D - rho rho^T) with D = diag(rho), and I_obs = I_c - I_m.
// I_m annihilates the constant, so each eigenvector can be taken with rho . v = 0, where
// I_c v = N D v: f is the largest eigenvalue of D^-1/2 I_m D^-1/2 / N, whose extra
// eigenvector sqrt(rho) has eigenvalue 0.
function missing(L, rho) {
  const M = infoMissing(L, rho).map((row, a) => row.map((x, b) => x / (N * Math.sqrt(rho[a] * rho[b]))));
  return Math.min(Math.max(...eigenvalues(M)), 1);
}

// Gibbs by data augmentation from rho: each step draws every object's bin from its posterior,
// then rho ~ Dir(1 + counts) (the Gibbs engine with the default prior).
function gibbs(L, rho, seed) {
  const u = rng(seed), chain = [];
  for (let t = 0; t < ITER; t++) {
    const counts = new Array(K).fill(1);
    for (const row of L) { // a bin with probability posterior(row, rho), without allocating it
      let s = 0, k = 0;
      for (let j = 0; j < K; j++) s += row[j] * rho[j];
      let x = u() * s;
      while (k < K - 1 && (x -= row[k] * rho[k]) > 0) k++;
      counts[k]++;
    }
    rho = dirichlet(counts, u);
    chain.push(rho);
  }
  return chain;
}

const lag1 = (x) => {
  const m = mean(x);
  let c0 = 0, c1 = 0;
  x.forEach((v, t) => { c0 += (v - m) ** 2; if (t) c1 += (v - m) * (x[t - 1] - m); });
  return c1 / c0;
};

export const title = "Missing information and Gibbs mixing";
export const controls = [
  { name: "width", label: "likelihood width (bins)", min: 0.05, max: 0.6, step: 0.05, value: 0.3 },
];

// One mock and its f at the true population for each of the slider's widths; its Gibbs chain
// when first shown. f is taken at the truth rather than at the most probable population,
// which needs a fit and puts some bins on the boundary of the simplex.
const { min, max, step } = controls[0];
const CASES = linspace(min, max, Math.round((max - min) / step) + 1).map((width) => {
  const { L } = mock(N, width * BIN, EDGES, 14, FLAT);
  return { width, L, rho: RHO, f: missing(L, RHO) };
});

export function compute({ width }) {
  const c = CASES[Math.round((width - min) / step)];
  if (!c.chain) {
    c.chain = gibbs(c.L, c.rho, 5);
    const ac = c.rho.map((_, k) => lag1(c.chain.map((r) => r[k])));
    c.bin = ac.indexOf(Math.max(...ac));
    c.lag1 = ac[c.bin];
  }
  return c;
}

export const readout = (state, { f, bin, lag1: a }) =>
  `f = ${fmt(f, 3)}, so τ ≈ ${+((1 + f) / (1 - f)).toPrecision(2)}; lag-one autocorrelation of ρ${sub(bin + 1)}: ${fmt(a, 2)}`;

export function draw({ width }, { f, chain, bin }) {
  const trace = chain.map((r) => r[bin]);
  const top = plot({ h: 200, edges: [0, max], ymax: 1, ylabel: "f", xlabel: "likelihood width (bins)",
    key: [["brand", "missing information f"]] });
  const low = plot({ h: 200, edges: [0, ITER], ymax: 1.15 * Math.max(...trace), ylabel: `ρ${sub(bin + 1)}`,
    xlabel: "Gibbs iteration", key: [["brand", `ρ${sub(bin + 1)}, the slowest bin`]] });
  const alt = `Top: the missing information rises from near 0 to near 1 as the likelihoods widen, ` +
    `with the width ${fmt(width, 2)} bins marked. Bottom: ${ITER} Gibbs draws of the slowest-mixing ` +
    `bin's fraction, which ${f > 0.9 ? "drift slowly" : f > 0.5 ? "wander" : "look like independent noise"}.`;
  return svg(top.h + low.h, alt, top.line(CASES.map((c) => c.width), CASES.map((c) => c.f), "brand") +
    top.line([width, width], [f, f], "mark") + top.axes() +
    `<g transform="translate(0,${top.h})">` + low.line(trace.map((_, t) => t + 1), trace, "brand") + low.axes() + "</g>");
}
