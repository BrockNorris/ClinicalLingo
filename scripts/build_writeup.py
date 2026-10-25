"""Build docs/architecture.png and docs/writeup.pdf without installing anything.

Needs: the `markdown` package importable (pip install markdown, or pip install --target <dir> markdown
and pass --mdlib <dir>), and Microsoft Edge (headless) on the PATH or at its default location.
Mermaid is rendered in the browser from a CDN at build time; nothing is uploaded anywhere.

    python scripts/build_writeup.py [--mdlib DIR] [--edge PATH]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EDGE_DEFAULTS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                 r"C:\Program Files\Microsoft\Edge\Application\msedge.exe", "msedge", "microsoft-edge"]
MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"

CSS = """
body{font-family:Calibri,'Segoe UI',Arial,sans-serif;font-size:10pt;line-height:1.3;max-width:7.4in;margin:0 auto;color:#111}
h1{font-size:15pt;margin:0 0 4pt} h2{font-size:12pt;margin:10pt 0 3pt;border-bottom:1px solid #999} h3{font-size:10.5pt;margin:7pt 0 2pt}
p{margin:3pt 0} ul,ol{margin:2pt 0 2pt 16pt} li{margin:1pt 0}
pre{font-family:Consolas,monospace;font-size:7.6pt;line-height:1.25;background:#f4f4f4;border:1px solid #ddd;padding:4pt;white-space:pre-wrap;word-break:break-word;margin:3pt 0}
code{font-family:Consolas,monospace;font-size:8.6pt} img{max-width:100%;max-height:4.6in;display:block;margin:4pt auto}
table{border-collapse:collapse;font-size:8.6pt} td,th{border:1px solid #bbb;padding:2pt 4pt;vertical-align:top}
@page{size:Letter;margin:0.55in}
"""


def find_edge(explicit: str | None) -> str:
    for cand in ([explicit] if explicit else []) + EDGE_DEFAULTS:
        if cand and (Path(cand).exists() or cand in ("msedge", "microsoft-edge")):
            return cand
    sys.exit("Edge not found; pass --edge <path to msedge.exe>")


def edge(edge_bin: str, profile: Path, *args: str) -> None:
    cmd = [edge_bin, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={profile}",
           "--virtual-time-budget=20000", "--run-all-compositor-stages-before-draw", *args]
    subprocess.run(cmd, check=True, timeout=180, capture_output=True)


def mermaid_source() -> str:
    return re.search(r"```mermaid\n(.*?)```", (DOCS / "architecture.md").read_text(encoding="utf-8"), re.S).group(1)


def render_diagram(edge_bin: str, profile: Path) -> Path:
    """Two headless passes: measure the rendered SVG, then screenshot at exactly that size."""
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>body{{margin:16px;background:#fff}}#d{{width:1800px}}</style></head>
<body><pre id="d" class="mermaid">{mermaid_source()}</pre>
<script src="{MERMAID_CDN}"></script><script>
mermaid.initialize({{startOnLoad:false,theme:'default',flowchart:{{htmlLabels:true,curve:'basis'}}}});
mermaid.run().then(()=>{{const r=document.querySelector('svg').getBoundingClientRect();
  document.body.setAttribute('data-size', Math.ceil(r.width+32)+'x'+Math.ceil(r.height+32));}});
</script></body></html>"""
    page = DOCS / "architecture.html"
    page.write_text(html, encoding="utf-8")
    dom = subprocess.run([edge_bin, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={profile}",
                          "--virtual-time-budget=20000", "--dump-dom", page.as_uri()],
                         check=True, timeout=180, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    m = re.search(r'data-size="(\d+)x(\d+)"', dom)
    if not m:
        sys.exit("mermaid did not render (is the CDN reachable?)")
    w, h = m.groups()
    out = DOCS / "architecture.png"
    edge(edge_bin, profile, f"--window-size={w},{h}", "--hide-scrollbars", f"--screenshot={out}", page.as_uri())
    page.unlink(missing_ok=True)
    return out


def build_pdf(edge_bin: str, profile: Path) -> Path:
    import markdown  # noqa: WPS433 - optional dependency, see module docstring

    md = (DOCS / "writeup.md").read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["fenced_code", "tables", "sane_lists"])
    # Embed the diagram as a vector: replace the PNG reference with the live Mermaid block.
    body = re.sub(r'<p><img[^>]*architecture\.png[^>]*></p>',
                  f'<div class="diagram"><pre class="mermaid">{mermaid_source()}</pre></div>', body, count=1)
    html = (f'<!doctype html><html><head><meta charset="utf-8"><title>ClinicalLingo Prototype 1</title><style>{CSS}'
            '.diagram{text-align:center;margin:4pt 0} .diagram svg{max-width:100%;max-height:4.9in;height:auto}</style></head>'
            f'<body>{body}<script src="{MERMAID_CDN}"></script>'
            "<script>mermaid.initialize({startOnLoad:true,theme:'default',flowchart:{htmlLabels:true,curve:'basis'}});</script></body></html>")
    page = DOCS / "writeup.html"
    page.write_text(html, encoding="utf-8")
    out = DOCS / "writeup.pdf"
    edge(edge_bin, profile, "--no-pdf-header-footer", f"--print-to-pdf={out}", page.as_uri())
    page.unlink(missing_ok=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdlib", help="folder containing the 'markdown' package (pip install --target)")
    ap.add_argument("--edge", help="path to msedge.exe")
    ap.add_argument("--skip-diagram", action="store_true")
    a = ap.parse_args()
    if a.mdlib:
        sys.path.insert(0, a.mdlib)
    edge_bin = find_edge(a.edge)
    with tempfile.TemporaryDirectory(prefix="cl-edge-") as tmp:
        profile = Path(tmp)
        if not a.skip_diagram:
            print("diagram ->", render_diagram(edge_bin, profile))
        print("pdf     ->", build_pdf(edge_bin, profile))
    return 0


if __name__ == "__main__":
    sys.exit(main())
