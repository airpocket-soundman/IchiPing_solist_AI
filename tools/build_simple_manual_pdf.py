"""Print the three-section quick manual as a three-page A4 portrait PDF.

Requires Chrome, BeautifulSoup4 and PyMuPDF. No device access.
"""
from pathlib import Path
import shutil
import subprocess

import pymupdf as fitz
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/manual/simple.html'
TMP = ROOT / '.local/simple-manual-pdf'
OUT = ROOT / 'output/pdf/ichiping_simple_a4.pdf'
TMP.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
soup = BeautifulSoup(SOURCE.read_text(encoding='utf-8'), 'html.parser')
soup.head.insert(0, soup.new_tag('base', href=SOURCE.parent.as_uri() + '/'))
# Explicit internal targets keep the PDF portable after moving it to another PC.
for a in soup.select('a[href="./"]'):
    a['href'] = 'https://airpocket-soundman.github.io/IchiPing_solist_AI/manual/'
source = TMP / 'source.html'
source.write_text(str(soup), encoding='utf-8')
raw = TMP / 'raw.pdf'
if raw.exists():
    raw.unlink()
chrome = Path('C:/Program Files/Google/Chrome/Application/chrome.exe')
subprocess.run([str(chrome), '--headless', '--disable-gpu', '--no-first-run',
                '--no-default-browser-check', '--no-pdf-header-footer',
                '--allow-file-access-from-files', '--virtual-time-budget=2000',
                '--user-data-dir=' + str(TMP / 'chrome-profile'),
                '--print-to-pdf=' + str(raw), source.as_uri()],
               check=True, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
doc = fitz.open(raw)
assert len(doc) == 3, f'Expected 3 pages, got {len(doc)}; adjust print layout.'
titles = ['01 セッティング', '02 推論', '03 ODL（現地校正）']
for i, page in enumerate(doc):
    assert abs(page.rect.width - 595.28) < 1 and abs(page.rect.height - 841.89) < 1
    assert ['セッティング', '推論', 'ODL（現地校正）'][i] in page.get_text()
    page.insert_text((40, 819), 'IchiPing on Solist-AI / Quick manual / EdgeTech+ 2026',
                     fontsize=8, color=(.35,.43,.41))
    page.insert_textbox(fitz.Rect(480,808,555,827), f'{i+1} / 3', fontsize=8, align=2,
                        color=(.35,.43,.41))
doc.set_toc([[1, title, i+1] for i,title in enumerate(titles)])
doc.set_metadata({'title':'IchiPing on Solist-AI 簡易マニュアル', 'author':'IchiPing',
                  'subject':'セッティング・推論・ODL / A4 portrait / 2026-11-16'})
doc.save(OUT, garbage=4, deflate=True)
doc.close()
shutil.copyfile(OUT, ROOT / 'docs/manual/ichiping_simple_a4.pdf')
print(f'Created 3-page A4 portrait PDF: {OUT}')
