// V-7: Dirichlet draws on the three-bin simplex, a triangle whose corners each put the whole
// population in one bin. Small concentrations push the draws into the corners, large ones
// pull them to the centre.
import { dirichlet, fmt, rng, sub, svg } from "./core.mjs";

const N = 1500, SIDE = 300, X0 = 50, Y0 = 300; // the base runs from (X0, Y0) to (X0 + SIDE, Y0)
const CORNERS = [[X0, Y0], [X0 + SIDE, Y0], [X0 + SIDE / 2, Y0 - (SIDE * Math.sqrt(3)) / 2]];

export const title = "Dirichlet draws on the simplex";
export const controls = [{ name: "loga", label: "log₁₀ α", min: -1, max: 1.5, step: 0.05, value: 0 }];

// N draws from Dir(alpha, alpha, alpha), the same seed for every alpha.
export function compute({ loga }) {
  const alpha = 10 ** loga, u = rng(7);
  return { alpha, draws: Array.from({ length: N }, () => dirichlet([alpha, alpha, alpha], u)) };
}

// The standard deviation of one fraction over the draws, and under Dir(alpha): each fraction
// is Beta(alpha, 2 alpha), with variance (1/3)(2/3) / (3 alpha + 1).
export function readout(state, { alpha, draws }) {
  const x = draws.map((d) => d[0]), m = x.reduce((a, b) => a + b, 0) / N;
  const sd = Math.sqrt(x.reduce((a, v) => a + (v - m) ** 2, 0) / (N - 1));
  return `α = ${+alpha.toPrecision(2)}: SD of each fraction ${fmt(sd, 3)} (Dirichlet ${fmt(Math.sqrt(2 / 9 / (3 * alpha + 1)), 3)})`;
}

export function draw({ loga }, { draws }) {
  const xy = (d) => [0, 1].map((i) => d.reduce((s, r, k) => s + r * CORNERS[k][i], 0).toFixed(1));
  const dots = draws.map((d) => `M${xy(d).join(",")}h0`).join("");
  const label = (k, dx, dy, anchor) =>
    `<text x="${CORNERS[k][0] + dx}" y="${CORNERS[k][1] + dy}" text-anchor="${anchor}">ρ${sub(k + 1)} = 1</text>`;
  const where = loga < -0.15 ? "crowd into the corners and along the edges" : loga <= 0.15
    ? "fill the triangle evenly" : loga < 1 ? "form a cloud around the centre" : "form a tight cluster at the centre";
  const alt = `A triangle with ${N} points, each a population of three bins; the points ${where}.`;
  return svg(330, alt, `<path class="axis" d="M${CORNERS.map((c) => c.join(",")).join("L")}Z"/>` +
    `<path class="dots" d="${dots}"/>` + label(0, 0, 24, "middle") + label(1, 0, 24, "middle") + label(2, 0, -10, "middle"));
}
