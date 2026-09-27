"""Render the ProtoPedia / PV figures (HTML + SVG) to images with headless Chrome.

    python media/figures/render_figures.py            # all figures
    python media/figures/render_figures.py fig_odl    # only some

Output goes to docs/protopedia/ (the images published on the ProtoPedia page and used by
media/pv/make_pv.py). Photos are saved as JPEG, diagrams as PNG.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "docs" / "protopedia"

# name: (width, height, output file name)
FIGURES = {
    "fig_hero": (1600, 900, "fig_hero.jpg"),
    "fig_system": (1600, 960, "fig_system.png"),
    "fig_inside": (1600, 900, "fig_inside.jpg"),
    "fig_operation": (1600, 660, "fig_operation.jpg"),
    "fig_pipeline": (1600, 960, "fig_pipeline.png"),
    "fig_variation": (1600, 830, "fig_variation.png"),
    "fig_params": (1600, 900, "fig_params.png"),
    "fig_odl": (1600, 900, "fig_odl.png"),
    "fig_autocollect": (1600, 980, "fig_autocollect.png"),
    "fig_accuracy": (1600, 900, "fig_accuracy.png"),
    "fig_thumb": (1280, 720, "yt_thumbnail.jpg"),
}

CHROME_CANDIDATES = [
    r"C:/Program Files/Google/Chrome/Application/chrome.exe",
    r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "google-chrome", "chromium", "chrome",
]


def find_chrome():
    for c in CHROME_CANDIDATES:
        if os.path.isfile(c) or shutil.which(c):
            return c
    sys.exit("Chrome / Edge not found")


def render(chrome, name, w, h, out_name):
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "shot.png"
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=1", f"--window-size={w},{h}",
                        f"--screenshot={png}", (HERE / f"{name}.html").as_uri()],
                       check=True, capture_output=True)
        dst = OUT / out_name
        im = Image.open(png)
        if dst.suffix == ".jpg":
            im.convert("RGB").save(dst, quality=90)
        else:
            im.save(dst)
    print(f"{name} -> {dst.relative_to(HERE.parents[1])}")


if __name__ == "__main__":
    chrome = find_chrome()
    names = sys.argv[1:] or list(FIGURES)
    for n in names:
        render(chrome, n, *FIGURES[n])
