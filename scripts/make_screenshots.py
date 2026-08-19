"""Generate PNG screenshots for NB1-NB8 via nbconvert + Playwright.

Workflow per notebook:
  1. nbconvert --to html (with executed outputs)
  2. Playwright load HTML -> screenshot full page PNG
  3. Save to submission/screenshots/nb{N}_results.png

NB1 and NB2 still missing executed outputs — execute them headless first.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbconvert.exporters import HTMLExporter
from nbclient import NotebookClient
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks"
OUTDIR = REPO / "submission" / "screenshots"
TMPDIR = REPO / ".screenshot_tmp"


def find_nb(n: int) -> Path:
    matches = sorted(NOTEBOOKS.glob(f"{n:02d}_*.ipynb"))
    assert matches, f"no notebook starting with {n:02d}_"
    return matches[0]


def execute_in_place(nb_path: Path) -> None:
    """Execute notebook so all cells produce output (idempotent)."""
    nb = nbformat.read(nb_path, as_version=4)
    cells_no_output = [i for i, c in enumerate(nb.cells)
                       if c.cell_type == "code" and not c.get("outputs")]
    if not cells_no_output:
        return
    print(f"  executing {nb_path.name} "
          f"({len(cells_no_output)} cells need output)...")
    client = NotebookClient(
        nb, kernel_name="python3", timeout=600,
        # The notebook's `_setup` import is a sibling module — must run
        # from notebooks/ so it resolves. Matches what Jupyter does.
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    client.execute()
    nbformat.write(nb, nb_path)
    print(f"  done")


def render_html(nb_path: Path, html_path: Path) -> None:
    """nbconvert .ipynb -> .html with embedded outputs."""
    nb = nbformat.read(nb_path, as_version=4)
    exporter = HTMLExporter()
    body, _ = exporter.from_notebook_node(nb)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(body, encoding="utf-8")


def screenshot(html_path: Path, png_path: Path) -> None:
    """Playwright loads HTML and captures full-page PNG."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1100, "height": 900},
                                  device_scale_factor=2)
        page = ctx.new_page()
        page.goto(f"file:///{html_path}".replace("\\", "/"))
        # wait for any embedded images / MathJax to settle
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(500)
        page.screenshot(path=str(png_path), full_page=True)
        browser.close()


def main() -> None:
    TMPDIR.mkdir(exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for n in range(1, 9):
        nb = find_nb(n)
        print(f"NB{n}: {nb.name}")
        execute_in_place(nb)
        html_path = TMPDIR / f"nb{n}.html"
        png_path = OUTDIR / f"nb{n}_results.png"
        render_html(nb, html_path)
        screenshot(html_path, png_path)
        print(f"  -> {png_path}  ({png_path.stat().st_size // 1024} KB)")
    print("DONE")


if __name__ == "__main__":
    main()
