"""Execute the code of Markdown pages, in order, one namespace per page.

Usage: python docs/_scripts/run_snippets.py docs/user-guide/*.md docs/tutorials/*.md
Blocks fenced as ```python or ```{code-cell} python run; other fences do not.
"""

from __future__ import annotations

import re
import sys
import time
import traceback

import matplotlib

matplotlib.use("Agg")

FENCE = re.compile(r"^```(?:python|\{code-cell\} python)[^\n]*\n(.*?)^```\s*$", re.S | re.M)


def run_page(path: str) -> bool:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    ns: dict = {"__name__": "__main__"}
    ok = True
    t0 = time.time()
    for i, m in enumerate(FENCE.finditer(text)):
        code = m.group(1)
        # drop MyST cell options such as ":tags: [hide-input]"
        lines = code.splitlines(keepends=True)
        while lines and re.match(r"^:[\w-]+:", lines[0]):
            lines.pop(0)
        code = "".join(lines)
        try:
            exec(compile(code, f"{path}[block {i}]", "exec"), ns)
        except Exception:
            print(f"FAILED {path} block {i}:\n{code}")
            traceback.print_exc()
            ok = False
            break
    print(f"{'ok  ' if ok else 'FAIL'} {path} ({time.time() - t0:.1f} s)")
    return ok


if __name__ == "__main__":
    results = [run_page(p) for p in sys.argv[1:]]
    sys.exit(0 if all(results) else 1)
