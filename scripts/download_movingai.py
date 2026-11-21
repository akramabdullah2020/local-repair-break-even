"""Download the Moving AI grid benchmark maps used in the study into data/movingai/.

  python scripts/download_movingai.py

Source: https://movingai.com/benchmarks/grids.html
Cite: N. R. Sturtevant, "Benchmarks for Grid-Based Pathfinding," IEEE TCIAIG 4(2):144-148, 2012.
Only '.map' files are extracted (flat, no paths from the archive are trusted).
"""
import io
import os
import sys
import time
import urllib.request
import zipfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(BASE, 'data', 'movingai')
ARCHIVES = {
  'room': 'https://movingai.com/benchmarks/room/room-map.zip',
  'maze': 'https://movingai.com/benchmarks/maze/maze-map.zip',
  'street': 'https://movingai.com/benchmarks/street/street-map.zip',
}


def fetch(url, tries=3):
  for k in range(tries):
    try:
      req = urllib.request.Request(url, headers={'User-Agent': 'research-reproduction-script'})
      with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()
    except Exception as e:  # transient network / TLS hiccups: retry, never disable verification
      if k == tries - 1:
        raise
      print(f'  retry after error: {e}')
      time.sleep(3)


def main():
  for sub, url in ARCHIVES.items():
    out = os.path.join(DEST, sub)
    os.makedirs(out, exist_ok=True)
    if any(f.endswith('.map') for f in os.listdir(out)):
      print(f'{sub}: already present, skipping'); continue
    print(f'{sub}: downloading {url}')
    data = fetch(url)
    n = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
      for name in zf.namelist():
        base = os.path.basename(name)
        if not base.endswith('.map') or base in ('', '.', '..'):
          continue
        with zf.open(name) as src, open(os.path.join(out, base), 'wb') as dst:
          dst.write(src.read())
        n += 1
    print(f'{sub}: {n} maps')
  return 0


if __name__ == '__main__':
  sys.exit(main())
