"""Download the UCI benchmark dataset into ``data/``.

    python scripts/download_data.py

Source: UCI ML Repository #296 - Diabetes 130-US Hospitals for Years 1999-2008
(https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008)

The zip is ~3 MB and expands to a ~19 MB CSV, which is why it is git-ignored
instead of committed.
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
REQUIRED = ("diabetic_data.csv", "IDS_mapping.csv")

DEST = Path(__file__).resolve().parents[1] / "data"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)

    if all((DEST / name).is_file() for name in REQUIRED):
        print(f"already present in {DEST}")
        return 0

    archive = DEST / "uci_diabetes.zip"
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, archive)
    print(f"  {archive.stat().st_size / 1e6:.1f} MB")

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DEST)

    missing = [name for name in REQUIRED if not (DEST / name).is_file()]
    if missing:
        print(f"ERROR: expected {missing} in the archive", file=sys.stderr)
        return 1

    print("extracted:", ", ".join(REQUIRED))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
