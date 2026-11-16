"""Print the current HTML manual to A4 portrait and add folios/bookmarks.

Requires Chrome (or --browser), BeautifulSoup4 and PyMuPDF. No device access.
Run from any directory: python tools/build_manual_pdf.py
"""
import argparse
import re
import shutil
import subprocess
from pathlib import Path

import pymupdf as fitz
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/manual/index.html'
PUBLIC = 'https://airpocket-soundman.github.io/IchiPing_solist_AI/'
TMP = ROOT / '.local/manual-pdf'
OUT = ROOT / 'output/pdf/ichiping_manual_a4.pdf'

parser = argparse.ArgumentParser()
parser.add_argument('--browser', type=Path, default=Path('C:/Program Files/Google/Chrome/Application/chrome.exe'))
args = parser.parse_args()
TMP.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)

soup = BeautifulSoup(SOURCE.read_text(encoding='utf-8'), 'html.parser')
base = soup.new_tag('base', href=SOURCE.parent.as_uri() + '/')
soup.head.insert(0, base)
chapters = [(s['id'], s.select_one('.num').get_text(strip=True), s.h2.get_text(strip=True))
            for s in soup.select('section.chapter')]
contents = soup.new_tag('div', attrs={'class': 'pdf-contents'})
contents.append(BeautifulSoup('<h2>目次</h2><ol></ol>', 'html.parser'))
for anchor, num, title in chapters:
    li = soup.new_tag('li')
    a = soup.new_tag('a', href='#' + anchor)
    a.string = f'{num}  {title}'
    li.append(a)
    contents.ol.append(li)
contents.append(BeautifulSoup(f'<p>HTML版：<a href="{PUBLIC}manual/">{PUBLIC}manual/</a><br>外観写真は撮影待ちの枠で掲載しています。</p>', 'html.parser'))
soup.select_one('header.hero').insert_after(contents)
for a in soup.select('a[href]'):
    href = a['href']
    # Anchors remain internal PDF links; supporting pages resolve to public URLs.
    if href.startswith('../'):
        a['href'] = PUBLIC + href[3:]
    elif href == 'assets/screens/manifest.json':
        a['href'] = PUBLIC + 'manual/' + href
for p in soup.select('#quick > p'):
    if 'チェックはこのブラウザ内' in p.get_text():
        p.string = '開場前に上から確認し、チェック欄に記入してください。'
# Keep each ODL procedure and its screenshots on the same printed page.
odl = soup.select_one('#odl')
for heading in list(odl.find_all('h3', recursive=False)):
    if not heading.get_text(strip=True).startswith(('6.2', '6.3', '6.4')):
        continue
    group = soup.new_tag('div', attrs={'class': 'pdf-procedure'})
    heading.insert_before(group)
    running = soup.new_tag('div', attrs={'class': 'pdf-running'})
    running.string = '06 現地校正（ODL）'
    group.append(running)
    node = heading
    while node:
        following = node.next_sibling
        group.append(node.extract())
        if following and getattr(following, 'name', None) == 'h3':
            break
        node = following
for section, label in [('#operation', '05 通常の使い方'), ('#trouble', '08 トラブルシューティング')]:
    heading = soup.select_one(section).find('h3', recursive=False)
    heading['class'] = ['pdf-continue']
    running = soup.new_tag('div', attrs={'class': 'pdf-running pdf-new-page'})
    running.string = label
    heading.insert_before(running)
soup.footer.clear()
soup.footer.string = 'IchiPing on Solist-AI / EdgeTech+ 2026 展示・操作マニュアル / 2026年11月16日'
style = soup.new_tag('style')
style.string = '''
@media print {
 @page { size:A4 portrait; margin:15mm 14mm 18mm; }
 html { scroll-behavior:auto; }
 body { font-size:9.5pt; line-height:1.65; }
 main { width:auto; }
 .pdf-contents { margin:7mm 0 0; padding:5mm 0; border-top:1px solid #d5dfdc; }
 .pdf-contents h2 { font-size:14pt; }
 .pdf-contents ol { columns:2; list-style:none; padding:0; margin:3mm 0; }
 .pdf-contents li { break-inside:avoid; font-size:9pt; margin:0 0 2mm; }
 .pdf-contents p { font-size:8pt; color:#576c71; margin:3mm 0 0; }
 .pdf-procedure,.pdf-new-page { break-before:page; }
 .pdf-running { font-size:10pt; font-weight:bold; color:#076d69; padding-bottom:3mm; border-bottom:1px solid #d5dfdc; margin-bottom:4mm; }
 .hero h1 { font-size:23pt; }
 section.chapter { padding:3mm 0 0; }
 .chapter-head { margin-bottom:4mm; padding-bottom:3mm; }
 h2 { font-size:18pt; }
 h3 { font-size:12pt; margin:5mm 0 2mm; }
 p { margin:2mm 0 3mm; }
 .lead { font-size:11pt; }
 .note { padding:3mm 4mm; margin:4mm 0; }
 .steps>li { padding-bottom:4mm; }
 .steps>li>strong { font-size:10.5pt; }
 .steps>li:before { width:7mm;height:7mm;font-size:8pt; }
 .screen-grid { grid-template-columns:repeat(3,minmax(0,1fr)); gap:3mm; }
 .screen-grid figure { margin:0; padding:2mm; }
 .screen-grid img { max-width:49mm; }
 figure.screen img { image-rendering:pixelated; }
 .screen-grid figcaption { font-size:8pt; }
 .screen-grid figcaption strong { font-size:9pt; }
 .photo { min-height:35mm; padding:4mm; }
 .photo.wide { min-height:40mm; }
 .photo strong { font-size:11pt; }
 .photo p { font-size:8pt; }
 .two { grid-template-columns:repeat(2,minmax(0,1fr)); }
 .two > .table-wrap { min-width:0; }
 th,td { min-width:0; padding:2.4mm; }
 table { font-size:8.5pt; line-height:1.6; margin:3mm 0; }
 .checklist label { padding:2.3mm 0; font-size:9.5pt; }
 .diagram { display:block; max-height:94mm; object-fit:contain; }
 #trouble .two figure.screen img { max-width:55mm; }
 .card { padding:3mm 4mm; }
 .flow { gap:2mm; }
 .flow span { min-width:30mm; padding:2mm; font-size:8pt; }
 .flow b { font-size:9pt; }
 .source-list { font-size:9pt; }
 footer { margin-top:3mm; padding-top:3mm; }
}
'''
soup.head.append(style)
html = TMP / 'source.html'
html.write_text(str(soup), encoding='utf-8')
raw = TMP / 'raw.pdf'
if raw.exists():
    raw.unlink()
command = [str(args.browser), '--headless', '--disable-gpu', '--no-first-run',
           '--no-default-browser-check', '--no-pdf-header-footer',
           '--allow-file-access-from-files', '--run-all-compositor-stages-before-draw',
           '--virtual-time-budget=3000', '--user-data-dir=' + str(TMP / 'chrome-profile'),
           '--print-to-pdf=' + str(raw), html.as_uri()]
subprocess.run(command, check=True, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
assert raw.is_file(), 'Chrome did not create the PDF'
doc = fitz.open(raw)
toc = [[1, '表紙・目次', 1]]
for i, page in enumerate(doc):
    assert abs(page.rect.width - 595.28) < 1 and abs(page.rect.height - 841.89) < 1, page.rect
    text = page.get_text()
    for anchor, num, title in chapters:
        # The section title appears in the cover contents too; skip the cover.
        if i and title in text and not any(t[1] == f'{num} {title}' for t in toc):
            toc.append([1, f'{num} {title}', i + 1])
    page.insert_text((40, 819), 'IchiPing on Solist-AI / EdgeTech+ 2026', fontsize=8, color=(.35,.43,.41))
    page.insert_textbox(fitz.Rect(480, 808, 555, 827), f'{i+1} / {len(doc)}', fontsize=8, align=2, color=(.35,.43,.41))
doc.set_toc(toc)
cover = doc[0]
for _, title, page_number in toc[1:]:
    # Title text is in Japanese, folio digits use the built-in Latin font.
    short_title = title.split(' ', 1)[1]
    rects = cover.search_for(short_title)
    if rects:
        rect = rects[0]
        right = 283 if rect.x0 < 290 else 553
        value = str(page_number)
        x = right - fitz.get_text_length(value, fontsize=8)
        cover.insert_text((x, rect.y1 - 1), value, fontsize=8, color=(.35,.43,.41))
doc.set_metadata({'title':'IchiPing on Solist-AI セットアップ・操作マニュアル',
                  'author':'IchiPing', 'subject':'EdgeTech+ 2026 / A4 portrait / 2026-11-16'})
doc.save(OUT, garbage=4, deflate=True)
count = len(doc)
doc.close()
shutil.copyfile(OUT, ROOT / 'docs/manual/ichiping_manual_a4.pdf')
print(f'Created A4 portrait PDF: {OUT} ({count} pages)')
print('Published copy: docs/manual/ichiping_manual_a4.pdf')
