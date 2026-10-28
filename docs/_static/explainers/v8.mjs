// V-8: what four priors say about the bins together. Draws of the fractions on 16 bins over
// [0, 1], and the correlation between every two bins' fractions over many draws.
import { DRAWS, dirichlet, fmt, linspace, mean, normal, num, plot, polya, rng, softmax, svg } from "./core.mjs";

const K = 16, N = 2000, EDGES = linspace(0, 1, K + 1);

// LogisticRandomWalk(order=2, scale=tau, trend_scale=3): second differences of eta are
// N(0, tau^2), the minimum-norm eta with those differences being the penalised modes; the
// linear trend, the unpenalised mode left once the constant is dropped, has standard
// deviation 3 sqrt(K) on a unit vector.
const XHAT = (() => { const x = Array.from({ length: K }, (_, k) => k - (K - 1) / 2), n = Math.hypot(...x); return x.map((v) => v / n); })();
function walk(tau, u) {
  const eta = [0, 0];
  for (let k = 2; k < K; k++) eta.push(tau * normal(u) + 2 * eta[k - 1] - eta[k - 2]);
  const m = mean(eta), dot = eta.reduce((a, v, k) => a + v * XHAT[k], 0);
  const trend = 3 * Math.sqrt(K) * normal(u);
  return softmax(eta.map((v, k) => v - m + (trend - dot) * XHAT[k]));
}

// LogisticGP("matern32", amplitude=1, length_scale=ell, n_basis=64): eta is a sum of sines on a
// box three times the width of [0, 1], each scaled by the square root of the Matern-3/2 spectral
// density 4 3^1.5 ell / (3 + omega^2 ell^2)^2 at its frequency omega.
const J = 64, HALF = 1.5, CENTRES = EDGES.slice(1).map((b) => b - 0.5 / K);
const OMEGA = Array.from({ length: J }, (_, j) => (Math.PI * (j + 1)) / (2 * HALF));
const SINES = CENTRES.map((x) => OMEGA.map((w) => Math.sin(w * (x - 0.5 + HALF)) / Math.sqrt(HALF)));
function gp(ell, u) {
  const c = OMEGA.map((w) => Math.sqrt((4 * 3 ** 1.5 * ell) / (3 + w * w * ell * ell) ** 2) * normal(u));
  return softmax(SINES.map((row) => { let eta = 0; for (let j = 0; j < J; j++) eta += row[j] * c[j]; return eta; }));
}

// Each prior: its name, the symbol of its hyperparameter, whose value is base * 10^s, and a draw.
const PRIORS = [
  { name: "Dirichlet", symbol: "α", base: 1, draw: (a, u) => dirichlet(new Array(K).fill(a), u) },
  { name: "Pólya tree", symbol: "c", base: 1, draw: (c, u) => polya(4, c, u) }, // power 2, flat base
  { name: "random walk", symbol: "τ", base: 0.1, draw: walk },
  { name: "Gaussian process", symbol: "ℓ", base: 0.6, draw: gp },
];

// The hyperparameter, N draws of the prior chosen and the correlation matrix of the fractions.
const cache = new Map();
export function compute({ prior, s }) {
  const key = `${prior} ${s}`;
  if (!cache.has(key)) {
    const value = PRIORS[prior].base * 10 ** s, u = rng(8);
    const draws = Array.from({ length: N }, () => PRIORS[prior].draw(value, u));
    const m = EDGES.slice(1).map((_, k) => mean(draws.map((d) => d[k])));
    const cov = m.map(() => new Array(K).fill(0));
    for (const d of draws) {
      const x = d.map((v, i) => v - m[i]);
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) cov[i][j] += x[i] * x[j];
    }
    const corr = cov.map((row, i) => row.map((x, j) => x / Math.sqrt(cov[i][i] * cov[j][j])));
    cache.set(key, { value, shown: draws.slice(0, DRAWS.length), corr });
  }
  return cache.get(key);
}

// Mean correlation of the pairs of bins d apart, for d = 1 and for d of 8 or more.
const apart = (corr, near) => {
  const r = corr.flatMap((row, i) => row.filter((_, j) => (near ? j === i + 1 : j >= i + 8)));
  return mean(r);
};

export const controls = [
  { name: "prior", label: "prior", options: PRIORS.map((p) => p.name), value: 0 },
  { name: "s", label: "log₁₀ scale", min: -1, max: 1, step: 0.05, value: 0 },
];
export const title = "Priors on 16 bins: draws and correlations";

export const readout = ({ prior }, { value, corr }) =>
  `${PRIORS[prior].name}, ${PRIORS[prior].symbol} = ${+value.toPrecision(2)}: mean correlation of neighbouring bins ` +
  `${fmt(apart(corr, true), 2)}, of bins 8 or more apart ${fmt(apart(corr, false), 2)}`;

const CUTS = [-0.6, -0.2, 0.2, 0.6]; // the boundaries of the correlation map's colours, div0 to div4
const shade = (r) => CUTS.filter((c) => r > c).length;

export function draw({ prior }, { shown, corr }) {
  const p = plot({ h: 210, edges: EDGES, ymax: K * Math.max(...shown.flat()), ylabel: "density", key: DRAWS });
  const cell = 12, x0 = 58, y0 = p.h + 34, bar = x0 + K * cell + 40;
  let map = `<text x="${x0}" y="${y0 - 12}">correlation between bins</text>`;
  const cells = [...CUTS, 1].map(() => ""); // one path of squares per colour
  corr.forEach((row, i) => row.forEach((r, j) => { cells[shade(r)] += `M${x0 + j * cell},${y0 + i * cell}h${cell}v${cell}h-${cell}z`; }));
  map += cells.map((d, c) => (d ? `<path class="div${c}" d="${d}"/>` : "")).join("");
  [[0, "1"], [K - 1, String(K)]].forEach(([k, s]) => {
    map += `<text x="${x0 - 6}" y="${y0 + (k + 0.5) * cell + 5}" text-anchor="end">${s}</text>` +
      `<text x="${x0 + (k + 0.5) * cell}" y="${y0 + K * cell + 18}" text-anchor="middle">${s}</text>`;
  });
  const ticks = [1, ...CUTS.slice().reverse(), -1], step = (K * cell) / (ticks.length - 1);
  ticks.forEach((t, i) => { // a colour bar, 1 at the top
    if (i < ticks.length - 1) map += `<path class="div${CUTS.length - i}" d="M${bar},${num(y0 + i * step)}h16v${num(step)}h-16z"/>`;
    map += `<text x="${bar + 22}" y="${num(y0 + i * step + 5)}">${fmt(t, 1)}</text>`;
  });
  const alt = `Three draws of a ${PRIORS[prior].name} prior on ${K} bins, and a ${K} by ${K} map of the ` +
    `correlations between the bins' fractions over ${N} draws.`;
  return svg(y0 + K * cell + 28, alt, DRAWS.map(([cls], i) => p.masses(shown[i], cls)).join("") + p.axes() + map);
}
