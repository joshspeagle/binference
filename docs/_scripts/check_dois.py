"""Check that every DOI in docs/references.bib is registered.

Usage: python docs/_scripts/check_dois.py

Asks the DOI resolver's handle API, which answers without contacting the publisher, so
publishers that refuse automated requests do not hide a mistyped DOI. Exits with status 1
and lists the DOIs that are not registered.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BIB = Path(__file__).resolve().parents[1] / "references.bib"


def registered(doi: str) -> bool:
    url = "https://doi.org/api/handles/" + urllib.parse.quote(doi)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response).get("responseCode") == 1
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return False
        raise


def main() -> int:
    dois = re.findall(r"^\s*doi\s*=\s*\{([^}]+)\}", BIB.read_text(encoding="utf-8"), re.M)
    missing = [d for d in dois if not registered(d)]
    for d in missing:
        print(f"not registered: {d}")
    print(f"{len(dois) - len(missing)} of {len(dois)} DOIs registered")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
