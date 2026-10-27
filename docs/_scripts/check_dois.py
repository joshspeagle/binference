"""Check that every DOI in docs/references.bib is registered.

Usage: python docs/_scripts/check_dois.py

Asks the DOI resolver's handle API, which answers without contacting the publisher, so
publishers that refuse automated requests do not hide a mistyped DOI. Exits with status 1
and lists the DOIs that are not registered; a DOI the resolver cannot answer for, after
one retry, is listed as unchecked and does not fail the run.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BIB = Path(__file__).resolve().parents[1] / "references.bib"


def registered(doi: str) -> bool | None:
    """Whether the resolver knows ``doi``; None when it could not answer."""
    url = "https://doi.org/api/handles/" + urllib.parse.quote(doi)
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response).get("responseCode") == 1
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return False
        except (urllib.error.URLError, TimeoutError):
            pass
        if attempt == 0:
            time.sleep(5)
    return None


def main() -> int:
    dois = re.findall(r"^\s*doi\s*=\s*\{([^}]+)\}", BIB.read_text(encoding="utf-8"), re.M)
    answers = {d: registered(d) for d in dois}
    for d, ok in answers.items():
        if not ok:
            print(f"{'not registered' if ok is False else 'unchecked'}: {d}")
    missing = sum(ok is False for ok in answers.values())
    print(f"{sum(ok is True for ok in answers.values())} of {len(dois)} DOIs registered")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
