// V-14: missing information and Gibbs mixing. The broader the likelihoods compared with the
// bins, the larger the missing information f, and the slower the Gibbs sampler mixes.
import { dirichlet, fmt, linspace, mock, normalise, plot, rng, sub, svg } from "./core.mjs";

const K = 10, N = 500, EDGES = linspace(0, 3, K + 1), BIN = EDGES[1] - EDGES[0], ITER = 400;
const WIDTHS = linspace(0.05, 1, 20); // the slider's positions, in bins

// The EM update of rho (PopulationModel.expected_counts(rho) / N), and the log-likelihood at
// rho. L is the likelihood matrix flattened, row by row, as in the functions below.
function em(L, rho) {
  const next = new Array(K).fill(0);
  let ll = 0;
  for (let i = 0; i < L.length; i += K) {
    let s = 0;
    for (let k = 0; k < K; k++) s += L[i + k] * rho[k];
    for (let k = 0; k < K; k++) next[k] += (L[i + k] * rho[k]) / s / N;
    ll += Math.log(s);
  }
  return [next, ll];
}

// The most probable population under the default flat Dirichlet (the maximum likelihood): EM
// from a flat start, accelerated by SQUAREM (Varadhan & Roland 2008) as MAP's EM is, until
// the log-likelihood gains less than 1e-9 per object.
function fit(L) {
  let v = new Array(K).fill(1 / K);
  for (let t = 0; t < 500; t++) {
    const [r1, f0] = em(L, v), [r2] = em(L, r1);
    const d1 = r1.map((x, k) => x - v[k]), d2 = r2.map((x, k) => x - 2 * r1[k] + v[k]);
    const a = Math.min(-Math.hypot(...d1) / (Math.hypot(...d2) || 1), -1);
    let cand = v.map((x, k) => x - 2 * a * d1[k] + a * a * d2[k]);
    if (!cand.every((x) => x >= 0)) cand = r2;
    const [next, f] = em(L, normalise(cand));
    v = f >= f0 ? next : r2; // an extrapolation that lowers the likelihood is dropped
    if (Math.abs(f - f0) < 1e-9 * N) break;
  }
  return v;
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

// The missing information f = 1 - min eig(I_obs, I_c) at rho (PopulationModel.missing_information),
// on the support of rho. In the log-fractions eta, I_c = N C with C = diag(rho) - rho rho^T and
// I_obs = I_c - I_m, with I_m the sum of the objects' diag(r_i) - r_i r_i^T. Both annihilate
// the constant, so all but one of the bins on the support give the same generalised
// eigenvalues as the isometric log-ratios: f is the largest eigenvalue of I_c^-1 I_m, here of
// G^-1 I_m G^-T with I_c = G G^T.
function missing(L, rho) {
  const top = Math.max(...rho), on = [...rho.keys()].filter((k) => rho[k] > 1e-9 * top), n = on.length - 1;
  if (n < 1) return 0;
  const Im = Array.from({ length: n }, () => new Array(n).fill(0)), r = new Array(n + 1);
  for (let i = 0; i < L.length; i += K) {
    let s = 0;
    for (let j = 0; j <= n; j++) { r[j] = L[i + on[j]] * rho[on[j]]; s += r[j]; }
    for (let j = 0; j <= n; j++) r[j] /= s;
    for (let a = 0; a < n; a++) {
      Im[a][a] += r[a];
      for (let b = 0; b < n; b++) Im[a][b] -= r[a] * r[b];
    }
  }
  const tot = on.reduce((a, k) => a + rho[k], 0), p = on.map((k) => rho[k] / tot);
  const G = Array.from({ length: n }, () => new Array(n).fill(0)); // Cholesky factor of I_c
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let x = N * ((i === j ? p[i] : 0) - p[i] * p[j]);
    for (let k = 0; k < j; k++) x -= G[i][k] * G[j][k];
    G[i][j] = i === j ? Math.sqrt(x) : x / G[j][j];
  }
  const solve = (b) => { // G^-1 b
    const y = [];
    for (let i = 0; i < n; i++) { let x = b[i]; for (let k = 0; k < i; k++) x -= G[i][k] * y[k]; y.push(x / G[i][i]); }
    return y;
  };
  const X = Im.map(solve); // rows: (G^-1 I_m)^T = I_m G^-T, since I_m is symmetric
  const T = X[0].map((_, j) => X.map((row) => row[j])); // G^-1 I_m
  const M = T.map(solve); // its columns: G^-1 I_m G^-T, symmetric
  return Math.min(Math.max(...eigenvalues(M)), 1);
}

// Gibbs by data augmentation from rho: each step draws every object's bin from its posterior,
// then rho ~ Dir(1 + counts) (the Gibbs engine with the default prior).
function gibbs(L, rho, seed) {
  const u = rng(seed), chain = [];
  for (let t = 0; t < ITER; t++) {
    const counts = new Array(K).fill(1);
    for (let i = 0; i < L.length; i += K) {
      let s = 0;
      for (let k = 0; k < K; k++) s += L[i + k] * rho[k];
      let x = u() * s, k = 0;
      while (k < K - 1 && (x -= L[i + k] * rho[k]) > 0) k++;
      counts[k]++;
    }
    rho = dirichlet(counts, u);
    chain.push(rho);
  }
  return chain;
}

const lag1 = (x) => {
  const m = x.reduce((a, b) => a + b, 0) / x.length;
  let c0 = 0, c1 = 0;
  x.forEach((v, t) => { c0 += (v - m) ** 2; if (t) c1 += (v - m) * (x[t - 1] - m); });
  return c1 / c0;
};

const cases = new Map(); // width -> its mock, fit and f
function at(width) {
  const key = width.toFixed(2);
  if (!cases.has(key)) {
    const { L } = mock(N, width * BIN, EDGES, 14, 0.5), flat = Float64Array.from(L.flat()), rho = fit(flat);
    cases.set(key, { L, flat, rho, f: missing(flat, rho) });
  }
  return cases.get(key);
}
const CURVE = WIDTHS.map((w) => at(w).f);

export const title = "Missing information and Gibbs mixing";
export const controls = [
  { name: "width", label: "likelihood width (bins)", min: 0.05, max: 1, step: 0.05, value: 0.3 },
];

export function compute({ width }) {
  const { L, flat, rho, f } = at(width), chain = gibbs(flat, rho, 5);
  const ac = rho.map((_, k) => lag1(chain.map((r) => r[k])));
  const bin = ac.indexOf(Math.max(...ac));
  return { L, rho, f, curve: CURVE, chain, bin, lag1: ac[bin] };
}

export const readout = (state, { f, bin, lag1: a }) =>
  `f = ${fmt(f, 2)}, so τ ≈ ${+((1 + f) / (1 - f)).toPrecision(2)}; lag-one autocorrelation of ρ${sub(bin + 1)}: ${fmt(a, 2)}`;

export function draw({ width }, { f, curve, chain, bin }) {
  const trace = chain.map((r) => r[bin]);
  const top = plot({ h: 200, edges: [0, 1], ymax: 1, ylabel: "f", xlabel: "likelihood width (bins)",
    key: [["brand", "missing information f"]] });
  const ymax = 1.15 * Math.max(...trace);
  const low = plot({ h: 200, edges: [0, ITER], ymax, ylabel: `ρ${sub(bin + 1)}`, xlabel: "Gibbs iteration",
    key: [["ink2", `ρ${sub(bin + 1)}, the slowest bin`]] });
  const alt = `Top: the missing information rises from near 0 to near 1 as the likelihoods widen, ` +
    `with the width ${fmt(width, 2)} bins marked. Bottom: ${ITER} Gibbs draws of the slowest-mixing ` +
    `bin's fraction, which ${f > 0.9 ? "drift slowly" : f > 0.5 ? "wander" : "look like independent noise"}.`;
  return svg(top.h + low.h, alt, top.line(WIDTHS, curve, "brand") + top.line([width, width], [f, f], "mark") + top.axes() +
    `<g transform="translate(0,${top.h})">` + low.line(trace.map((_, t) => t + 1), trace, "ink2") + low.axes() + "</g>");
}
