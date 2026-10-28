// Shared pieces of the docs' explainers: seeded random numbers, the normal CDF and an SVG plot.
// Every function is pure, so docs/_scripts/explainers.mjs can draw the still frames in Node.
// Colours are CSS classes (binference.css), so a frame follows the page's theme.

export const WIDTH = 400; // viewBox width; text is 16 units, 13.4 px when drawn 334 px wide

export function rng(seed) { // mulberry32: uniform numbers in (0, 1)
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
  };
}

export function normal(u) { // a standard normal draw from the uniform source u
  return Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u());
}

export function Phi(x) { // the standard normal CDF, through erfc (W. J. Cody's rational fit, error < 1.2e-7)
  const z = Math.abs(x) / Math.SQRT2, t = 1 / (1 + 0.5 * z);
  const erfc = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? 1 - erfc / 2 : erfc / 2;
}

export const fmt = (x, d) => (x < 0 ? "−" : "") + Math.abs(x).toFixed(d);
const num = (x) => +x.toFixed(2);
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");

// A plot of height h with data ranges x and y. Its methods return SVG strings.
export function plot({ h = 260, x, y, xlabel = "", ylabel = "", xticks = [], yticks = [] }) {
  const m = { l: ylabel ? 44 : 14, r: 12, t: 10, b: xlabel ? 48 : 28 };
  const sx = (v) => m.l + ((v - x[0]) / (x[1] - x[0])) * (WIDTH - m.l - m.r);
  const sy = (v) => h - m.b - ((v - y[0]) / (y[1] - y[0])) * (h - m.t - m.b);
  const p = {
    h, sx, sy,
    line(xs, ys, cls) {
      return `<path class="${cls}" d="M${xs.map((v, i) => `${num(sx(v))},${num(sy(ys[i]))}`).join("L")}"/>`;
    },
    stairs(edges, values, cls) { // a step at each value between its two edges, closed to the baseline
      let d = `M${num(sx(edges[0]))},${num(sy(y[0]))}`;
      values.forEach((v, k) => { d += `V${num(sy(v))}H${num(sx(edges[k + 1]))}`; });
      return `<path class="${cls}" d="${d}V${num(sy(y[0]))}"/>`;
    },
    vline(v, cls) { return `<path class="${cls}" d="M${num(sx(v))},${m.t}V${h - m.b}"/>`; },
    text(px, py, s, cls = "", anchor = "start") {
      return `<text class="${cls}" x="${num(px)}" y="${num(py)}" text-anchor="${anchor}">${esc(s)}</text>`;
    },
    key(px, py, items) { // a legend: a short line in each item's class, then its label
      return items.map(([cls, label], i) => {
        const yy = py + 20 * i;
        return `<path class="${cls}" d="M${px},${yy - 5}h22"/>` + p.text(px + 28, yy, label, "key");
      }).join("");
    },
    axes() {
      const base = sy(y[0]);
      let s = `<path class="axis" d="M${m.l},${m.t}V${base}H${WIDTH - m.r}"/>`;
      xticks.forEach((v) => { s += p.text(sx(v), base + 20, v, "tick", "middle"); });
      yticks.forEach((v) => { s += p.text(m.l - 6, sy(v) + 5, v, "tick", "end"); });
      if (xlabel) s += p.text((m.l + WIDTH - m.r) / 2, h - 8, xlabel, "label", "middle");
      if (ylabel) s += `<text class="label" transform="translate(16,${num((m.t + base) / 2)}) rotate(-90)" text-anchor="middle">${esc(ylabel)}</text>`;
      return s;
    },
  };
  return p;
}

// The whole frame: an svg element whose accessible name is the explainer's alt text.
export const svg = (h, alt, body) =>
  `<svg viewBox="0 0 ${WIDTH} ${h}" role="img" aria-label="${esc(alt).replace(/"/g, "&quot;")}">${body}</svg>`;

// Bin averages of a Gaussian likelihood N(d, sigma^2) on the given edges (Likelihoods.from_gaussian).
export const binAverages = (d, sigma, edges) =>
  edges.slice(1).map((b, k) => (Phi((b - d) / sigma) - Phi((edges[k] - d) / sigma)) / (b - edges[k]));

// One EM step for the population masses rho: the mean over objects of their posteriors
// L_ik rho_k / sum_j L_ij rho_j (PopulationModel.expected_counts(rho) / N).
export function emStep(L, rho) {
  const next = rho.map(() => 0);
  for (const row of L) {
    const w = row.map((l, k) => l * rho[k]), s = w.reduce((a, b) => a + b, 0);
    w.forEach((x, k) => { next[k] += x / s / L.length; });
  }
  return next;
}

// A mock catalogue: n true values from a two-peaked population on [0, 3], each measured with
// Gaussian error sigma, as the likelihood matrix on the edges.
export function mock(n, sigma, edges, seed) {
  const u = rng(seed), z = [];
  while (z.length < n) {
    const v = u() < 0.6 ? 0.9 + 0.25 * normal(u) : 2.1 + 0.3 * normal(u);
    if (v > 0 && v < 3) z.push(v);
  }
  return { z, L: z.map((v) => binAverages(v + sigma * normal(u), sigma, edges)) };
}

// The masses of that population in each bin (its two normal peaks, cut to [0, 3]).
export function truth(edges) {
  const cdf = (x) => 0.6 * Phi((x - 0.9) / 0.25) + 0.4 * Phi((x - 2.1) / 0.3);
  const m = edges.slice(1).map((b, k) => cdf(b) - cdf(edges[k])), s = m.reduce((a, b) => a + b, 0);
  return m.map((x) => x / s);
}

export const normalise = (w) => { const s = w.reduce((a, b) => a + b, 0); return w.map((x) => x / s); };
// The mean of z under bin masses m, each bin's mass at its centre.
export const meanZ = (m, edges) => m.reduce((s, x, k) => s + (x * (edges[k] + edges[k + 1])) / 2, 0);
export const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
export const density = (masses, edges) => masses.map((m, k) => m / (edges[k + 1] - edges[k]));
