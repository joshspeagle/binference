  const npdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  const POPS = {
    exp: (z) => Math.exp(-1.5 * z),
    peak: (z) => Math.exp(-0.5 * ((z - 1.2) / 0.1) ** 2) + 0.15,
    bimodal: (z) => 0.55 * npdf(z, 0.8, 0.35) + 0.45 * npdf(z, 2.1, 0.4),
    power: (z) => 1 / (z + 0.02),
  };
  function inv(A) {           // Gauss-Jordan inverse of a small symmetric matrix
    const n = A.length, M = A.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]]; const piv = M[c][c]; if (Math.abs(piv) < 1e-300) return null;
      for (let j = 0; j < 2 * n; j++) M[c][j] /= piv;
      for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c]; if (f) for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j]; }
    }
    return M.map(r => r.slice(n));
  }
  const ST = { key: "", res: null };
  function compute(pop, space, basis, K, sig) {
    const lo = space === "lin" ? 0 : 0.05, hi = 3;
    const edges = Array.from({ length: K + 1 }, (_, k) => space === "lin" ? lo + (hi - lo) * k / K : lo * Math.pow(hi / lo, k / K));
    const f = (pop === "power" && space === "log") ? (z) => 1 / z : POPS[pop];
    // z quadrature: Simpson, 40 intervals per bin
    const zq = [], wq = [], bq = [];
    for (let k = 0; k < K; k++) { const a = edges[k], b = edges[k + 1], m = 40, h = (b - a) / m; for (let j = 0; j <= m; j++) { zq.push(a + j * h); wq.push(h / 3 * ((j === 0 || j === m) ? 1 : (j % 2 ? 4 : 2))); bq.push(k); } }
    const Q = zq.length, fq = zq.map(f); let F = 0; for (let q = 0; q < Q; q++) F += fq[q] * wq[q];
    const mtrue = new Float64Array(K); for (let q = 0; q < Q; q++) mtrue[bq[q]] += fq[q] * wq[q] / F;
    // bin shapes phi_k(z_q) (sparse: list of [k, value] per node)
    const cen = edges.slice(0, K).map((a, k) => 0.5 * (a + edges[k + 1]));
    const phi = Array.from({ length: Q }, () => []);
    if (basis === "flatz") for (let q = 0; q < Q; q++) phi[q].push([bq[q], 1 / (edges[bq[q] + 1] - edges[bq[q]])]);
    else if (basis === "flatlog") for (let q = 0; q < Q; q++) { const k = bq[q]; phi[q].push([k, 1 / (zq[q] * Math.log(edges[k + 1] / edges[k]))]); }
    else {
      const raw = Array.from({ length: Q }, () => []), norm = new Float64Array(K);
      for (let q = 0; q < Q; q++) {
        const z = zq[q];
        if (z <= cen[0]) raw[q].push([0, 1]); else if (z >= cen[K - 1]) raw[q].push([K - 1, 1]);
        else { let j = 0; while (cen[j + 1] < z) j++; const t = (z - cen[j]) / (cen[j + 1] - cen[j]); raw[q].push([j, 1 - t], [j + 1, t]); }
        raw[q].forEach(([k, v]) => { norm[k] += v * wq[q]; });
      }
      for (let q = 0; q < Q; q++) phi[q] = raw[q].map(([k, v]) => [k, v / norm[k]]);
    }
    const Mm = Array.from({ length: K }, () => new Float64Array(K));   // bin masses from weights
    for (let q = 0; q < Q; q++) phi[q].forEach(([k, v]) => { Mm[bq[q]][k] += v * wq[q]; });
    // data density on a d grid, and L_k(d)
    const D = 400, d0 = lo - 4 * sig, d1 = hi + 4 * sig, dd = (d1 - d0) / (D - 1);
    const L = Array.from({ length: D }, () => new Float64Array(K)), wd = new Float64Array(D); let wt = 0;
    for (let i = 0; i < D; i++) {
      const d = d0 + i * dd; let p = 0;
      for (let q = 0; q < Q; q++) { const kq = npdf(d, zq[q], sig) * wq[q]; p += kq * fq[q]; for (const [k, v] of phi[q]) L[i][k] += kq * v; }
      wd[i] = p / F * dd; wt += wd[i];
    }
    for (let i = 0; i < D; i++) wd[i] /= wt;
    const gfun = (r) => { const g = new Float64Array(K); let obj = 0; for (let i = 0; i < D; i++) { let s = 0; for (let k = 0; k < K; k++) s += L[i][k] * r[k]; obj += wd[i] * Math.log(s); for (let k = 0; k < K; k++) g[k] += wd[i] * L[i][k] / s; } return { g, obj }; };
    const Fm = (r) => { const { g } = gfun(r); const o = r.map((v, k) => v * g[k]); const t = o.reduce((a, b) => a + b, 0); return o.map(v => v / t); };
    let r = new Float64Array(K).fill(1 / K), it = 0, conv = false;
    for (it = 0; it < 1500; it++) {        // SQUAREM-accelerated EM on the large-N objective
      const r1 = Fm(r), r2 = Fm(r1), d1v = r1.map((v, k) => v - r[k]), d2v = r2.map((v, k) => v - 2 * r1[k] + r[k]);
      const n1 = Math.hypot(...d1v), n2 = Math.hypot(...d2v);
      let nxt = r2;
      if (n2 > 0) {
        const a = Math.min(-n1 / n2, -1); let c = r.map((v, k) => Math.max(v - 2 * a * d1v[k] + a * a * d2v[k], 0)); const t = c.reduce((p, q) => p + q, 0);
        if (t > 0) { c = Fm(c.map(v => v / t)); if (gfun(c).obj >= gfun(r2).obj) nxt = c; }
      }
      const ch = Math.max(...nxt.map((v, k) => Math.abs(v - r[k]))); r = nxt; if (ch < 1e-11) { conv = true; break; }
    }
    // Fisher information per object and the covariance of the bin masses
    const { g } = gfun(r);
    const I = Array.from({ length: K }, () => new Float64Array(K));
    for (let i = 0; i < D; i++) { let s = 0; for (let k = 0; k < K; k++) s += L[i][k] * r[k]; const w2 = wd[i] / (s * s); for (let j = 0; j < K; j++) for (let k = j; k < K; k++) I[j][k] += w2 * L[i][j] * L[i][k]; }
    for (let j = 0; j < K; j++) for (let k = 0; k < j; k++) I[j][k] = I[k][j];
    const A = (j, a) => (j < K - 1 ? (j === a ? 1 : 0) : -1);                // rho_K = 1 - sum of the others
    const Ip = Array.from({ length: K - 1 }, (_, a) => Array.from({ length: K - 1 }, (_, b) => { let s = 0; for (let j = 0; j < K; j++) for (let k = 0; k < K; k++) s += A(j, a) * I[j][k] * A(k, b); return s; }));
    const tr = Ip.reduce((s, row, a) => s + row[a], 0); Ip.forEach((row, a) => { row[a] += 1e-9 * tr; });
    const Ci = inv(Ip);
    const C = Array.from({ length: K }, (_, j) => Array.from({ length: K }, (_, k) => { let s = 0; for (let a = 0; a < K - 1; a++) for (let b = 0; b < K - 1; b++) s += A(j, a) * Ci[a][b] * A(k, b); return s; }));
    const mhat = Mm.map(row => row.reduce((s, v, k) => s + v * r[k], 0)); const mt = mhat.reduce((a, b) => a + b, 0);
    const varm = Mm.map(row => { let s = 0; for (let j = 0; j < K; j++) for (let k = 0; k < K; k++) s += row[j] * C[j][k] * row[k]; return Math.max(s, 0); });
    return { edges, mtrue: Array.from(mtrue), mhat: mhat.map(v => v / mt), varm, conv, f, F, lo, hi, gap: Math.max(...g) - 1 };
  }

const fmt=(R)=>{let nc=Infinity;R.mtrue.forEach((m,k)=>{const b=R.mhat[k]-m;if(m>1e-3&&Math.abs(b)>0)nc=Math.min(nc,R.varm[k]/(b*b));});return nc.toExponential(1)+(R.conv?"":" (unconverged)");};
for (const [pop,space,K,sig] of [["exp","lin",10,0.15],["exp","lin",10,0.05],["bimodal","lin",10,0.15],["peak","lin",15,0.1],["power","log",10,0.05],["exp","log",10,0.05]]) {
  const row=["flatz","flatlog","hat"].map(b=>b+": "+fmt(compute(pop,space,b,K,sig)));
  console.log(pop,space,"K="+K,"sig="+sig,"|",row.join("  "));
}
