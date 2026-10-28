// Shared pieces of the docs' explainers: seeded random numbers, the normal CDF, the model's
// sums (bin averages, an object's posterior, EM), a mock catalogue and an SVG plot.
// Every function is pure, so docs/_scripts/explainers.mjs can draw the still frames in Node.
// Colours are CSS classes (binference.css), so a frame follows the page's theme.

// The viewBox width. Text is 16 units: 13.4 px when drawn 334 px wide, the readable width of a
// 390 px phone, above figure-text (12 px; docs/_scripts/brand.py).
const WIDTH = 400;

export function rng(seed) { // mulberry32: uniform numbers in (0, 1)
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
  };
}

export const normal = (u) => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u());

// The logarithm of a Gamma(a, 1) draw (Marsaglia & Tsang), shapes below one boosted in log
// space so that small concentrations do not underflow.
function logGamma(a, u) {
  if (a < 1) return logGamma(a + 1, u) + Math.log(u()) / a;
  const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do { x = normal(u); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    if (Math.log(u()) < 0.5 * x * x + d - d * v + d * Math.log(v)) return Math.log(d * v);
  }
}

// A Dir(alpha) draw: the softmax of log-gamma variates, floored at 1e-300 (_dirichlet_draw).
export function dirichlet(alpha, u) {
  const g = alpha.map((a) => logGamma(a, u)), top = Math.max(...g);
  return normalise(g.map((x) => Math.exp(x - top))).map((x) => Math.max(x, 1e-300));
}

// A draw of the 2^levels bin fractions of a Polya tree with a flat base measure (PolyaTree):
// each split at depth h (the root has h = 1) is Dir(c h^power, c h^power).
export function polya(levels, c, u, power = 2) {
  let rho = [1];
  for (let h = 1; h <= levels; h++) {
    const a = c * h ** power;
    rho = rho.flatMap((r) => { // a Beta(a, a) share, from two log-gamma variates
      const v = 1 / (1 + Math.exp(logGamma(a, u) - logGamma(a, u)));
      return [Math.max(r * v, 1e-300), Math.max(r * (1 - v), 1e-300)];
    });
  }
  return rho;
}

function Phi(x) { // the standard normal CDF, through erfc (W. J. Cody's rational fit, error < 1.2e-7)
  const z = Math.abs(x) / Math.SQRT2, t = 1 / (1 + 0.5 * z);
  const erfc = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? 1 - erfc / 2 : erfc / 2;
}

export const fmt = (x, d) => (x < 0 ? "−" : "") + Math.abs(x).toFixed(d);
// A number in subscript digits, for labels such as ρ₄.
export const sub = (n) => String(n).replace(/\d/g, (d) => "₀₁₂₃₄₅₆₇₈₉"[d]);
const num = (x) => +x.toFixed(2);
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");

export const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
export const normalise = (w) => { const s = w.reduce((a, b) => a + b, 0); return w.map((x) => x / s); };
// The mean of z under bin masses m, each bin's mass at its centre.
export const meanZ = (m, edges) => m.reduce((s, x, k) => s + (x * (edges[k] + edges[k + 1])) / 2, 0);

// Bin averages of a Gaussian likelihood N(d, sigma^2) on the given edges (Likelihoods.from_gaussian).
export const binAverages = (d, sigma, edges) =>
  edges.slice(1).map((b, k) => (Phi((b - d) / sigma) - Phi((edges[k] - d) / sigma)) / (b - edges[k]));

// An object's posterior masses r_k = L_k rho_k / sum_j L_j rho_j (PopulationModel.responsibilities).
export const posterior = (row, rho) => normalise(row.map((l, k) => l * rho[k]));

// EM from a flat population: element t is the population after t steps, each step the mean of
// the objects' posteriors (PopulationModel.expected_counts(rho) / N). Element 1 is the stack.
export function emPath(L, steps) {
  const K = L[0].length, w = new Float64Array(K), path = [new Array(K).fill(1 / K)];
  for (let t = 0; t < steps; t++) {
    const rho = path[t], next = new Array(K).fill(0);
    for (const row of L) { // posterior(row, rho), without allocating it
      let s = 0;
      for (let k = 0; k < K; k++) { w[k] = row[k] * rho[k]; s += w[k]; }
      for (let k = 0; k < K; k++) next[k] += w[k] / s / L.length;
    }
    path.push(next);
  }
  return path;
}

// The mock population: two normal peaks, cut to [0, 3].
const PEAKS = [[0.6, 0.9, 0.25], [0.4, 2.1, 0.3]]; // weight, centre, width

// n true values from the mock population, a share flat of them drawn uniformly on [0, 3]
// instead, each measured with Gaussian error sigma, as the likelihood matrix on the edges.
export function mock(n, sigma, edges, seed, flat = 0) {
  const u = rng(seed), z = [];
  while (z.length < n) {
    if (flat && u() < flat) { z.push(3 * u()); continue; }
    const [, c, w] = u() < PEAKS[0][0] ? PEAKS[0] : PEAKS[1], v = c + w * normal(u);
    if (v > 0 && v < 3) z.push(v);
  }
  return { z, L: z.map((v) => binAverages(v + sigma * normal(u), sigma, edges)) };
}

// The mock population's masses in each bin (without a flat share).
export function truth(edges) {
  const cdf = (x) => PEAKS.reduce((s, [a, c, w]) => s + a * Phi((x - c) / w), 0);
  return normalise(edges.slice(1).map((b, k) => cdf(b) - cdf(edges[k])));
}

// Round tick values from 0 to at least max: at most five, at steps of 1, 2, 2.5 or 5 times a power of 10.
function ticks(max) {
  const p = 10 ** Math.floor(Math.log10(max / 4));
  const step = [1, 2, 2.5, 5, 10].map((s) => s * p).find((s) => max / s <= 4);
  return Array.from({ length: Math.ceil(max / step - 1e-9) + 1 }, (_, i) => +(i * step).toFixed(6));
}

// A plot over the bins with the given edges, from 0 to ymax, with its legend (key: [class, label]
// pairs, two to a row) in a band above. h is the frame's height without that band, p.h the whole.
// stairs draws values per bin; masses draws masses per bin as densities.
export function plot({ h = 250, edges, ymax, ylabel, key, xlabel = "true value z" }) {
  const x = [edges[0], edges[edges.length - 1]], xticks = ticks(x[1]);
  const yticks = ticks(ymax), top = yticks[yticks.length - 1], band = 22 * Math.ceil(key.length / 2);
  const m = { l: 58, r: 16, t: 12 + band, b: 48 };
  h += band;
  const sx = (v) => m.l + ((v - x[0]) / (x[1] - x[0])) * (WIDTH - m.l - m.r);
  const sy = (v) => h - m.b - (Math.min(v, top) / top) * (h - m.t - m.b);
  const text = (px, py, s, anchor) =>
    `<text x="${num(px)}" y="${num(py)}" text-anchor="${anchor}">${esc(s)}</text>`;
  const p = {
    h,
    line(xs, ys, cls) {
      return `<path class="${cls}" d="M${xs.map((v, i) => `${num(sx(v))},${num(sy(ys[i]))}`).join("L")}"/>`;
    },
    stairs(values, cls) { // a step at each value across its bin, closed to the baseline
      let d = `M${num(sx(edges[0]))},${num(sy(0))}`;
      values.forEach((v, k) => { d += `V${num(sy(v))}H${num(sx(edges[k + 1]))}`; });
      return `<path class="${cls}" d="${d}V${num(sy(0))}"/>`;
    },
    masses(ms, cls) { return p.stairs(ms.map((v, k) => v / (edges[k + 1] - edges[k])), cls); },
    vline(v, cls) { return `<path class="${cls}" d="M${num(sx(v))},${m.t}V${h - m.b}"/>`; },
    axes() { // the axes, their ticks and labels, and the legend
      const base = sy(0);
      let s = `<path class="axis" d="M${m.l},${m.t}V${base}H${WIDTH - m.r}"/>`;
      xticks.forEach((v) => { s += text(sx(v), base + 20, v, "middle"); });
      yticks.forEach((v) => { s += text(m.l - 6, sy(v) + 5, v, "end"); });
      s += text((m.l + WIDTH - m.r) / 2, h - 8, xlabel, "middle");
      s += `<text transform="translate(16,${num((m.t + base) / 2)}) rotate(-90)" text-anchor="middle">${esc(ylabel)}</text>`;
      key.forEach(([cls, label], i) => {
        const kx = 14 + (i % 2) * 196, ky = 18 + 22 * Math.floor(i / 2);
        s += `<path class="${cls}" d="M${kx},${ky - 5}h22"/>` +
          `<text class="key" x="${kx + 28}" y="${ky}">${esc(label)}</text>`;
      });
      return s;
    },
  };
  return p;
}

// The whole frame: an svg element whose accessible name is the explainer's alt text.
export const svg = (h, alt, body) => `<svg viewBox="0 0 ${WIDTH} ${h}" role="img" aria-label="${esc(alt)}">${body}</svg>`;
