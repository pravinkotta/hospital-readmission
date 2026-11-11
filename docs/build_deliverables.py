# Builds the Week-9 deliverables from markdown:
#   docs/writeup.md -> Capstone-Writeup.pdf   (A4 portrait, exactly 1 page)
#   docs/deck.md     -> Capstone-Deck.pdf      (16:9 landscape, exactly 10 slides)
import html
import os
import re
import subprocess
import sys
import urllib.parse

import markdown

BASE = os.path.dirname(os.path.abspath(__file__))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

BASE_CSS = """
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Arial, sans-serif; color: #16202c; margin: 0; }
p { margin: 4pt 0; }
strong { color: #0d3f8f; }
a { color: #0d3f8f; text-decoration: none; }
code { font-family: Consolas, "Courier New", monospace; font-size: .92em;
  background: #eef1f5; padding: 0 3px; border-radius: 2px; }
ul, ol { margin: 3pt 0; padding-left: 16pt; }
li { margin: 2pt 0; }
table { border-collapse: collapse; margin: 5pt 0; font-size: .92em; }
th { background: #0d3f8f; color: #fff; text-align: left; padding: 3px 7px; font-weight: 600; }
td { border-bottom: 1px solid #d5dde7; padding: 3px 7px; vertical-align: top; }
tr:nth-child(even) td { background: #f3f6fb; }
blockquote { margin: 5pt 0; padding: 4pt 9pt; background: #fff7ea;
  border-left: 4px solid #e0a83c; color: #5a4200; font-size: .95em; }
blockquote p { margin: 0; }
pre { background: #f4f6f9; border: 1px solid #d5dde7; border-radius: 4px;
  padding: 6pt 8pt; font-family: Consolas, monospace; font-size: .8em;
  line-height: 1.3; white-space: pre-wrap; margin: 5pt 0; }
pre code { background: none; padding: 0; }
h1, h2, h3 { color: #0d3f8f; page-break-after: avoid; }
hr { border: none; border-top: 1px solid #c9d4e2; margin: 6pt 0; }
"""

WRITEUP_CSS = """
@page { size: A4; margin: 9mm 10mm; }
body { font-size: 8.9pt; line-height: 1.36; }
h1 { font-size: 15pt; margin: 0 0 2pt; line-height: 1.15; }
.wu-sub { font-size: 8.8pt; color: #444; margin-bottom: 6pt;
  padding-bottom: 4pt; border-bottom: 1.5px solid #0d3f8f; }
.wu-sub strong { color: #0d3f8f; }
h2 { font-size: 10.2pt; margin: 8pt 0 2.5pt; padding: 3pt 7pt;
  background: #eaf1fd; border-left: 4px solid #0d3f8f; }
table { font-size: 8.3pt; margin: 4pt 0; width: 100%; }
th, td { padding: 2.5px 7px; }
pre { font-size: 7.5pt; padding: 4pt 6pt; }
"""

REPORT_CSS = """
@page { size: A4; margin: 12mm 11mm; }
body { font-size: 9.2pt; line-height: 1.42; }
h1 { font-size: 17pt; margin: 0 0 2pt; line-height: 1.15; }
.rp-sub { font-size: 8.8pt; color: #444; margin-bottom: 7pt; padding-bottom: 4pt;
  border-bottom: 1.5px solid #0d3f8f; }
h2 { font-size: 11.5pt; margin: 11pt 0 3pt; padding: 3pt 7pt; background: #eaf1fd;
  border-left: 4px solid #0d3f8f; page-break-after: avoid; }
h3 { font-size: 9.8pt; margin: 8pt 0 2pt; color: #0d3f8f; page-break-after: avoid; }
table { font-size: 8.4pt; margin: 5pt 0; width: 100%; }
th, td { padding: 3px 6px; }
pre { font-size: 7.6pt; }
li { margin: 1.5pt 0; }
p { margin: 3pt 0; }
"""

DECK_CSS = """
@page { size: 297mm 167mm; margin: 0; }
body { font-size: 16.5pt; line-height: 1.38; }
.slide { width: 297mm; height: 167mm; padding: 13mm 17mm 12mm;
  page-break-after: always; overflow: hidden; position: relative; }
.slide:last-child { page-break-after: auto; }
.slide h1 { font-size: 33pt; margin: 0 0 3mm; line-height: 1.1;
  border-bottom: 2.2mm solid #0d3f8f; padding-bottom: 3mm; }
.slide h2 { font-size: 18pt; color: #333; font-weight: normal; margin: 0 0 5mm; }
.slide h3 { font-size: 17pt; color: #0d3f8f; margin: 5mm 0 2mm; }
.slide li { margin: 2.4pt 0; }
.slide table { font-size: 14.5pt; }
.slide th, .slide td { padding: 4px 12px; }
.slide pre { font-size: 12.5pt; }
.slide .foot { position: absolute; bottom: 7mm; left: 17mm; right: 17mm;
  font-size: 11pt; color: #7b8794; border-top: 1px solid #d5dde7;
  padding-top: 2.5mm; display: flex; justify-content: space-between; }
/* title slide */
.slide.title-slide { display: flex; flex-direction: column; justify-content: center;
  text-align: left; background: linear-gradient(135deg, #f2f7ff 0%, #ffffff 60%); }
.slide.title-slide h1 { font-size: 44pt; border: none; margin-bottom: 2mm; }
.slide.title-slide h2 { font-size: 21pt; color: #0d3f8f; margin-bottom: 9mm; }
.slide.title-slide li { font-size: 17pt; margin: 3.5pt 0; }
"""


def to_html(md_text: str) -> str:
    return markdown.markdown(
        md_text,
        extensions=["fenced_code", "tables", "sane_lists", "nl2br"],
        output_format="html5",
    )


def render_pdf(html_text: str, pdf_path: str) -> None:
    # the HTML intermediate lives next to the source in docs/, the PDF at repo root
    html_path = os.path.join(BASE, os.path.basename(pdf_path)[:-4] + ".html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    url = "file:///" + urllib.parse.quote(html_path.replace("\\", "/"))
    cmd = [
        EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
        "--virtual-time-budget=15000", f"--print-to-pdf={pdf_path}", url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if not os.path.exists(pdf_path):
        print(f"PDF FAILED: {pdf_path}")
        print(result.stdout[-1500:])
        print(result.stderr[-1500:])


def page_count(pdf_path: str) -> int:
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        return len(doc)


def build_writeup() -> None:
    src = os.path.join(BASE, "writeup.md")
    out = os.path.join(BASE, "..", "Capstone-Writeup.pdf")
    raw = open(src, encoding="utf-8").read()
    # split off the header block (title + meta) for custom rendering
    header, rest = raw.split("\n---\n", 1)
    lines = [line for line in header.strip().split("\n") if line.strip()]
    title = lines[0].lstrip("# ").strip()
    # the meta line is markdown (**...**); render it as HTML links/bold
    meta = to_html(lines[1]) if len(lines) > 1 else ""
    meta = re.sub(r"</?p>", "", meta)
    body = re.sub(r"^\s*---\s*$", "", rest, flags=re.M)  # drop section rules
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Write-up</title><style>{BASE_CSS}{WRITEUP_CSS}</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="wu-sub">{meta}</div>
{to_html(body)}
</body></html>"""
    render_pdf(page, os.path.abspath(out))
    n = page_count(os.path.abspath(out))
    print(f"write-up: {n} page(s) -> {os.path.abspath(out)}")
    if n != 1:
        print("WARNING: write-up must be exactly 1 page — adjust WRITEUP_CSS")


def build_deck() -> None:
    src = os.path.join(BASE, "deck.md")
    out = os.path.join(BASE, "..", "Capstone-Deck.pdf")
    raw = open(src, encoding="utf-8").read()
    blocks = [b.strip() for b in re.split(r"\n---\s*\n", raw) if b.strip()]
    if len(blocks) != 10:
        print(f"WARNING: deck has {len(blocks)} slides, expected 10")

    slides = []
    for index, block in enumerate(blocks, start=1):
        body = to_html(block)
        classes = "slide title-slide" if index == 1 else "slide"
        if index == 1:
            # title slide: promote headings, drop the auto h2 duplication
            body = body.replace("<h1>", "<h1>").replace("<h2>", "<h2>")
        footer = (
            f'<div class="foot"><span>Hospital Readmission Risk — AI Career Compass capstone</span>'
            f"<span>{index} / {len(blocks)}</span></div>"
        )
        slides.append(f'<section class="{classes}">{body}{footer}</section>')

    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Capstone deck</title><style>{BASE_CSS}{DECK_CSS}</style></head><body>
{"".join(slides)}
</body></html>"""
    render_pdf(page, os.path.abspath(out))
    n = page_count(os.path.abspath(out))
    print(f"deck: {n} slides -> {os.path.abspath(out)}")
    if n != len(blocks):
        print("WARNING: slide count mismatch — check for overflow")


def build_report() -> None:
    src = os.path.join(BASE, "project-report.md")
    out = os.path.join(BASE, "..", "Project-Report.pdf")
    raw = open(src, encoding="utf-8").read()
    header, rest = raw.split("\n---\n", 1)
    lines = [line for line in header.strip().split("\n") if line.strip()]
    title = lines[0].lstrip("# ").strip()
    sub = "<br>".join(to_html(line) for line in lines[1:])
    sub = re.sub(r"</?p>", "", sub)
    body = re.sub(r"^\s*---\s*$", "", rest, flags=re.M)  # drop section rules
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Project report</title><style>{BASE_CSS}{REPORT_CSS}</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="rp-sub">{sub}</div>
{to_html(body)}
</body></html>"""
    render_pdf(page, os.path.abspath(out))
    n = page_count(os.path.abspath(out))
    print(f"report: {n} pages -> {os.path.abspath(out)}")
    if n < 8:
        print("WARNING: report looks too short — check for render errors")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "writeup"):
        build_writeup()
    if which in ("all", "deck"):
        build_deck()
    if which in ("all", "report"):
        build_report()
