"""Fetch real discharge-guideline documents into guidelines/ for the RAG index.

Run once, offline afterwards: the fetched markdown is committed to the repo so
the index can be rebuilt without network access (embeddings still need
OPENAI_API_KEY). Each file records its source URL for citation.

    python scripts/fetch_guidelines.py
"""

from __future__ import annotations

import os
import re
import urllib.request
from html.parser import HTMLParser

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "guidelines")

# AHRQ Re-Engineered Discharge (RED) Toolkit — public-domain US Government work.
SOURCES = [
    (
        "ahrq-red-tool-1-overview.md",
        "AHRQ RED Toolkit — Tool 1: Overview of the Re-Engineered Discharge",
        "https://www.ahrq.gov/patient-safety/settings/hospital/red/toolkit/redtool1.html",
    ),
    (
        "ahrq-red-tool-3-assess-readiness.md",
        "AHRQ RED Toolkit — Tool 3: Assess Patient Readiness for Discharge",
        "https://www.ahrq.gov/patient-safety/settings/hospital/red/toolkit/redtool3.html",
    ),
    (
        "ahrq-red-tool-5-followup.md",
        "AHRQ RED Toolkit — Tool 5: Post-Discharge Follow-Up and Reinforcement",
        "https://www.ahrq.gov/patient-safety/settings/hospital/red/toolkit/redtool5.html",
    ),
    (
        "ahrq-red-toolkit-index.md",
        "AHRQ Re-Engineered Discharge (RED) Toolkit — Index",
        "https://www.ahrq.gov/patient-safety/settings/hospital/red/toolkit/index.html",
    ),
]


class TextExtractor(HTMLParser):
    """Crude but dependency-free HTML -> text extractor for article bodies."""

    SKIP = {"script", "style", "nav", "header", "footer", "form", "svg"}
    BLOCK = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "br", "ul", "ol", "table"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0
        self._in_article = False

    def handle_starttag(self, tag, attrs) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
        if tag in ("article", "main"):
            self._in_article = True
        if tag in self.BLOCK:
            self.parts.append("\n")
        if tag in ("h1", "h2", "h3"):
            self.parts.append("\n## ")
        if tag == "li":
            self.parts.append("\n- ")

    def handle_endtag(self, tag) -> None:
        if tag in self.SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in ("article", "main"):
            self._in_article = False
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    text = "".join(parser.parts)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_main(html: str) -> str:
    """Keep only the <main>/<article> region when present (drops site chrome)."""
    m = re.search(r"<(main|article)\b[^>]*>(.*?)</\1>", html, re.S | re.I)
    return m.group(2) if m else html


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (research; RAG corpus builder)")]
    for filename, title, url in SOURCES:
        dest = os.path.join(OUT_DIR, filename)
        try:
            with opener.open(url, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"FAIL {url}: {exc}")
            continue
        body = html_to_text(extract_main(html))
        # Trim common chrome left in place when no <main> matched
        if len(body) > 100:
            doc = (
                f"# {title}\n\n"
                f"> Source: {url}\n"
                f"> Retrieved via scripts/fetch_guidelines.py (AHRQ, public domain).\n\n"
                f"{body}\n"
            )
            with open(dest, "w", encoding="utf-8") as f:
                f.write(doc)
            print(f"OK   {filename} ({len(doc)//1024} KB)")
        else:
            print(f"FAIL {url}: extracted only {len(body)} chars")


if __name__ == "__main__":
    main()
