#!/usr/bin/env python3
"""Clean jagged edges on a pixel-art sprite.

Two conservative passes on the alpha silhouette:
  1. drop orphan pixels (opaque, no opaque 4-neighbours)
  2. fill notches (transparent, >=3 opaque 4-neighbours) with the
     most common neighbouring colour
Passes repeat until nothing changes.

usage: dejaggy.py input.png output.png
"""
import sys
from collections import Counter
from PIL import Image

N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def neighbours(px, w, h, x, y):
    return [px[x + dx, y + dy] for dx, dy in N4
            if 0 <= x + dx < w and 0 <= y + dy < h]


def dejaggy(im):
    im = im.convert("RGBA")
    w, h = im.size
    px = im.load()
    orphans = notches = 0
    while True:
        changes = []
        for y in range(h):
            for x in range(w):
                p = px[x, y]
                opaque = [n for n in neighbours(px, w, h, x, y) if n[3] > 0]
                if p[3] > 0 and not opaque:
                    changes.append((x, y, (0, 0, 0, 0)))
                    orphans += 1
                elif p[3] == 0 and len(opaque) >= 3:
                    colour = Counter(opaque).most_common(1)[0][0]
                    changes.append((x, y, colour))
                    notches += 1
        if not changes:
            break
        for x, y, c in changes:
            px[x, y] = c
    return im, orphans, notches


if __name__ == "__main__":
    src, dst = sys.argv[1:3]
    out, o, n = dejaggy(Image.open(src))
    out.save(dst)
    print(f"removed {o} orphan pixels, filled {n} notches -> {dst}")
