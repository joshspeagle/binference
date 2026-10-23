  function erf(x) { const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x); return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)); }
  const Phi = (x) => 0.5 * (1 + erf(x / Math.SQRT2)), npdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  const gaussOf = (rnd) => () => { let u = 0; while (u === 0) u = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd()); };
  function mle(L, N, K, iters) {
    const r = new Float64Array(K).fill(1 / K);
    for (let t = 0; t < iters; t++) {
      const g = new Float64Array(K);
      for (let i = 0; i < N; i++) { let s = 0; for (let k = 0; k < K; k++) s += L[i * K + k] * r[k]; if (s <= 0) continue; for (let k = 0; k < K; k++) g[k] += L[i * K + k] / s; }
      let tot = 0; for (let k = 0; k < K; k++) { r[k] *= g[k]; tot += r[k]; } for (let k = 0; k < K; k++) r[k] /= tot;
    }
    return Array.from(r);
  }
  const tv = (a, b) => 0.5 * a.reduce((s, v, k) => s + Math.abs(v - b[k]), 0);
  // ---------- §3 posteriors explorer
  const QK = 12, QE = Array.from({ length: QK + 1 }, (_, k) => 0.05 * Math.pow(60, k / QK)), QV = QE.slice(0, QK).map((a, k) => QE[k + 1] - a);
  const QPOP = (z) => Math.exp(-0.5 * (Math.log(z / 0.7) / 0.45) ** 2) / z + 0.6 * Math.exp(-0.5 * (Math.log(z / 1.9) / 0.15) ** 2) / z;   // densities in z
  const QINT = { bump: (z) => Math.exp(-0.5 * ((z - 1.6) / 0.4) ** 2), flat: () => 1, falling: (z) => Math.exp(-z) };
  function binMass(f, a, b) { let s = 0; const m = 60, h = (b - a) / m; for (let j = 0; j <= m; j++) s += f(a + j * h) * ((j === 0 || j === m) ? 1 : (j % 2 ? 4 : 2)); return s * h / 3; }
  const QTRUE = (() => { const m = QE.slice(0, QK).map((a, k) => binMass(QPOP, a, QE[k + 1])); const t = m.reduce((a, b) => a + b, 0); return m.map(v => v / t); })();
  const QSx = { key: "", res: null };
  function qCompute(intName, seed) {
    const rnd = mulberry32(seed * 3571 + 11), gauss = gaussOf(rnd), N = 3000;
    const fint = QINT[intName], Pi = QE.slice(0, QK).map((a, k) => binMass(fint, a, QE[k + 1])), pit = Pi.reduce((a, b) => a + b, 0); for (let k = 0; k < QK; k++) Pi[k] /= pit;
    const piDens = Pi.map((p, k) => p / QV[k]);
    const Lc = new Float64Array(N * QK), Lb = new Float64Array(N * QK), Ld = new Float64Array(N * QK), Le = new Float64Array(N * QK);
    for (let i = 0; i < N; i++) {
      // draw z exactly from the population: a mixture of two log-normals (weights ∝ 0.45 and 0.6 × 0.15), truncated to the bins
      let z; do { z = rnd() < 0.45 / 0.54 ? Math.exp(Math.log(0.7) + 0.45 * gauss()) : Math.exp(Math.log(1.9) + 0.15 * gauss()); } while (z < QE[0] || z > QE[QK]);
      const sig = 0.08 * (1 + z), d = z + sig * gauss();
      const Ltrue = QE.slice(0, QK).map((a, k) => (Phi((QE[k + 1] - d) / sig) - Phi((a - d) / sig)) / QV[k]);
      const Pm = Ltrue.map((l, k) => l * QV[k] * Pi[k] / QV[k]);   // posterior mass ∝ L_k · Π_k
      for (let k = 0; k < QK; k++) {
        Lc[i * QK + k] = Pm[k] / Pi[k];                  // correct
        Lb[i * QK + k] = Pm[k];                          // posteriors used as likelihoods
        Ld[i * QK + k] = Pm[k] / piDens[k];              // interim density read as mass (volume factor left behind)
        Le[i * QK + k] = Pm[k] / QV[k] / Pi[k];          // posterior density read as mass
      }
    }
    return { Pi, fits: [mle(Lc, N, QK, 600), mle(Lb, N, QK, 600), mle(Ld, N, QK, 600), mle(Le, N, QK, 600)] };
  }
  function gaussLegendre(n) {      // nodes on [0, 1] and weights summing to one
    const x = [], w = [];
    for (let i = 1; i <= n; i++) {
      let t = Math.cos(Math.PI * (i - 0.25) / (n + 0.5)), dp = 0;
      for (let it = 0; it < 100; it++) { let p0 = 1, p1 = t; for (let k = 2; k <= n; k++) { const p2 = ((2 * k - 1) * t * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; } const pn = n === 1 ? t : p1, pm = n === 1 ? 1 : p0; dp = n * (t * pn - pm) / (t * t - 1); const dt = pn / dp; t -= dt; if (Math.abs(dt) < 1e-15) break; }
      x.push(0.5 * (t + 1)); w.push(1 / ((1 - t * t) * dp * dp));
    }
    return { x, w };
  }
  const REF = gaussLegendre(20);   // reference: 32 panels x 20 points per bin
  const RQ = (() => {
    const ratios = Array.from({ length: 60 }, (_, j) => Math.pow(10, -1.3 + 2.0 * j / 59)), out = {};
    for (const n of [1, 3, 5, 9]) {
      const g = n === 1 ? { x: [0.5], w: [1] } : gaussLegendre(n);
      out[n] = ratios.map(sr => {
        let worst = 0;
        for (let off = 0; off <= 0.5; off += 0.02) {           // measurement position within bin 0 (bin width 1)
          const d = off, vals = [];
          for (let k = -6; k <= 6; k++) { let ex = 0; for (let p = 0; p < 32; p++) REF.x.forEach((u, j) => { ex += REF.w[j] / 32 * npdf(k + (p + u) / 32, d, sr); }); let ap = 0; g.x.forEach((u, j) => { ap += g.w[j] * npdf(k + u, d, sr); }); vals.push([ex, ap]); }
          const mx = Math.max(...vals.map(v => v[0]));
          vals.forEach(([ex, ap]) => { if (ex > 0.01 * mx) worst = Math.max(worst, Math.abs(ap - ex) / ex); });
        }
        return [sr, worst];
      });
    }
    return out;
  })();

const at=(n,s)=>RQ[n].reduce((b,p)=>Math.abs(Math.log(p[0]/s))<Math.abs(Math.log(b[0]/s))?p:b)[1];
console.log("quad worst rel err  n=3: h/10",at(3,0.1).toPrecision(2)," h/4",at(3,0.25).toPrecision(2)," h",at(3,1).toPrecision(2),"| n=5: h/4",at(5,0.25).toPrecision(2)," h",at(5,1).toPrecision(2));
for (const intn of ["bump","flat","falling"]) { const r=qCompute(intn,1); console.log(intn, r.fits.map(f=>tv(f,QTRUE).toFixed(3)).join("  ")); }
