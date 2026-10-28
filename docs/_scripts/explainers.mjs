// Draws the still frame of every explainer in docs/_static/explainers into docs/_ext/explainers.json,
// which the {explainer} directive inlines so a page is complete before JavaScript runs.
//
//   node docs/_scripts/explainers.mjs            write the frames
//   node docs/_scripts/explainers.mjs --check    fail if the committed frames are stale (CI)
//   node docs/_scripts/explainers.mjs --compute v1 '{"sigma": 0.1, "width": 0.25}'
//                                                print an explainer's numbers (tests/test_explainers.py)
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const dir = new URL("../_static/explainers/", import.meta.url);
const out = fileURLToPath(new URL("../_ext/explainers.json", import.meta.url));
const load = (id) => import(new URL(`${id}.mjs`, dir));
const [mode, id, state] = process.argv.slice(2);

if (mode === "--compute") {
  process.stdout.write(JSON.stringify((await load(id)).compute(JSON.parse(state))));
} else {
  const frames = {};
  for (const name of readdirSync(dir).filter((f) => /^v\d+\.mjs$/.test(f)).sort()) {
    const ex = await load(name.slice(0, -4));
    const start = Object.fromEntries(ex.controls.map((c) => [c.name, c.value]));
    frames[name.slice(0, -4)] = { title: ex.title, controls: ex.controls, svg: ex.draw(start), readout: ex.readout(start) };
  }
  const text = JSON.stringify(frames, null, 1) + "\n";
  if (mode === "--check") {
    let old = "";
    try { old = readFileSync(out, "utf8"); } catch {}
    if (old !== text) {
      console.error("docs/_ext/explainers.json is stale: run node docs/_scripts/explainers.mjs");
      process.exit(1);
    }
  } else {
    writeFileSync(out, text);
  }
}
