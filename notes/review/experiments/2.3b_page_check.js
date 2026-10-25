const { chromium } = require('/opt/node-tools/node_modules/playwright');
(async () => {
  const b = await chromium.launch({ executablePath: require('fs').readdirSync('/opt/pw-browsers').includes('chromium-1194') ? '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' : undefined });
  const p = await b.newPage({ viewport: { width: 420, height: 5200 } });
  const errs = [];
  p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  p.on('pageerror', e => errs.push('PAGEERR ' + e.message));
  await p.goto('file://' + process.cwd() + '/host.html');
  await p.waitForTimeout(6000);
  const f = p.frames()[1];
  f.on && 0;
  const info = await f.evaluate(() => ({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth, stats: document.getElementById('k-stats').innerText, ppc: document.getElementById('k-ppc-stats').innerText, h: document.body.scrollHeight }));
  console.log(JSON.stringify(info), errs);
  const el = await f.$('#d6'); await el.screenshot({ path: 'expl.png' });
  // T = 0.5: slider log10 = -0.3
  await f.evaluate(() => { const e = document.getElementById('k-T'); e.value = -0.3; e.dispatchEvent(new Event('input')); });
  await p.waitForTimeout(3000);
  console.log('T=0.5', await f.evaluate(() => document.getElementById('k-stats').innerText));
  await f.evaluate(() => { const e = document.getElementById('k-T'); e.value = 0; e.dispatchEvent(new Event('input')); const s = document.getElementById('k-shift'); s.value = 1; s.dispatchEvent(new Event('input')); });
  await p.waitForTimeout(3000);
  console.log('shift=1', await f.evaluate(() => document.getElementById('k-stats').innerText + ' | ' + document.getElementById('k-ppc-stats').innerText));
  await (await f.$('#d6')).screenshot({ path: 'expl_shift.png' });
  for (const id of ['d1', 'd7']) await (await f.$('#' + id)).screenshot({ path: id + '.png' });
  await b.close();
})();
