"""Rotate and enhance a scan for reading, optionally cropping to a region.

Usage: uv run python ../data/enhance.py <input.jpg> <rotate_deg> <out.jpg> [left top right bottom as 0-1 fractions]
"""

import sys

from PIL import Image, ImageEnhance, ImageOps

src, rot, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
img = Image.open(src).convert("L")
if rot:
    img = img.rotate(rot, expand=True)
if len(sys.argv) == 8:
    l, t, r, b = map(float, sys.argv[4:8])
    w, h = img.size
    img = img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
img = ImageOps.autocontrast(img, cutoff=1)
img = ImageEnhance.Sharpness(img).enhance(1.6)
img.save(out, quality=92)
print(out, img.size)
