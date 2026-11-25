"""Build report/report.pdf from report/report.md (Markdown -> HTML -> headless Chrome/Edge PDF).

Usage (repository root): python bench/make_tables.py && python report/build_pdf.py
"""
import shutil
import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
CSS = """
@font-face { font-family: 'Outfit'; src: url('fonts/Outfit.ttf'); font-weight: 100 900; }
@font-face { font-family: 'Arimo'; src: url('fonts/Arimo-400.ttf'); font-weight: 400; }
@font-face { font-family: 'Arimo'; src: url('fonts/Arimo-600.ttf'); font-weight: 600; }
@page { size: A4; margin: 14mm 0 14mm 0; }
@page :first { margin-top: 0; }
* { box-sizing: border-box; }
body { font-family: 'Arimo', 'Helvetica Neue', Arial, sans-serif; font-size: 9.6pt; line-height: 1.5; color: #1b2430; margin: 0; }
.content { padding: 0 16mm; }
.cover { position: relative; overflow: hidden; z-index: 0; padding: 22mm 16mm 16mm 16mm; margin: 0 -16mm 6mm -16mm; }
.kicker { font-size: 9pt; letter-spacing: 1.5px; text-transform: uppercase; color: #0b2a5b; font-weight: 600; }
h1.title { font-family: 'Outfit', sans-serif; font-size: 46pt; font-weight: 400; letter-spacing: -1.5px; color: #0b1b33; margin: 6px 0 4px 0; border: none; }
.star { font-size: 26pt; vertical-align: top; margin-left: 4px; color: #0b2a5b; }
.subtitle { font-size: 10.5pt; color: #24364f; }
h2 { font-family: 'Outfit', sans-serif; font-size: 18pt; font-weight: 400; color: #0b1b33; background: #dceafc; border-bottom: 2px solid #0b2a5b;
     padding: 7px 12px 7px 50px; margin: 26px 0 10px 0; position: relative; page-break-after: avoid; }
h2::before { content: "↘"; position: absolute; left: 12px; top: 5px; font-size: 20pt; color: #0b2a5b; }
h3 { font-family: 'Outfit', sans-serif; font-size: 12pt; font-weight: 600; color: #0b2a5b; margin: 16px 0 6px 0; page-break-after: avoid; }
p { margin: 6px 0; }
ul { margin: 4px 0 8px 0; padding-left: 18px; }
li { margin: 2px 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px 0; font-size: 8.8pt; page-break-inside: avoid; }
th { text-align: left; font-weight: 600; border-top: 1px solid #0b2a5b; border-bottom: 1px solid #0b2a5b; padding: 5px 8px; }
td { border-bottom: 1px solid #c9d6e8; padding: 4px 8px; vertical-align: top; }
code { font-family: Consolas, monospace; font-size: 8.4pt; background: #eef4fd; padding: 0 3px; border-radius: 3px; }
.stats { display: flex; gap: 8px; margin: 12px 0 14px 0; }
.stat { flex: 1; position: relative; overflow: hidden; z-index: 0; color: #fff; border-radius: 10px; padding: 12px 12px; }
.stat .num { font-family: 'Outfit', sans-serif; font-size: 22pt; font-weight: 300; letter-spacing: -0.5px; }
.stat .lbl { font-size: 8pt; opacity: 0.9; }
.figs { display: flex; gap: 6px; margin: 10px 0; page-break-inside: avoid; }
figure { flex: 1; margin: 0; text-align: center; }
figure img { width: 100%; border-radius: 8px; border: 1px solid #c9d6e8; }
figcaption { font-size: 7.8pt; color: #4a5a70; margin-top: 3px; }
strong { color: #0b1b33; }
img.bg { position: absolute; left: 0; top: 0; width: 100%; height: 100%; z-index: -1; border: none; border-radius: 0; margin: 0; }
img.chart { display: block; margin: 8px 0 12px 0; page-break-inside: avoid; }
.row { display: flex; gap: 14px; align-items: center; page-break-inside: avoid; }
.note { flex: 1; font-size: 9.4pt; color: #24364f; border-left: 3px solid #2f6fd1; padding-left: 10px; }
"""


def main() -> None:
    md = (HERE / "report.md").read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "md_in_html"])
    html = HERE / "report.html"
    html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>PS5 Pharma Inspector report</title><style>{CSS}</style></head><body><div class='content'>{body}</div></body></html>", encoding="utf-8")
    browser = next((b for b in (shutil.which("chrome"), r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe") if b and Path(b).exists()), None)
    if browser is None:
        raise SystemExit("Chrome or Edge not found; open report/report.html and print to PDF")
    out = HERE / "report.pdf"
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}", html.as_uri()], check=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
