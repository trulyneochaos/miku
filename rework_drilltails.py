#!/usr/bin/env python3
"""Rework the outline of the twin drill tails.

Finds the hair by colour (cyan / pink / yellow families), takes the
area within a few pixels of it inside each tail's bounding box, and
inside that area only:
  1. smooths the silhouette (stronger blur than the whole-sprite pass,
     since the curls are thick)
  2. thins the outline to a single pixel: pure-black pixels just behind
     the outer edge become the neighbouring hair colour
  3. re-draws the outline on every edge pixel

usage: rework_drilltails.py input.png output.png
"""
import colorsys
import os
import sys
from collections import Counter
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smooth_outline import (N4, OUTLINE, is_edge, repair_outline,  # noqa: E402
                            smooth_staircase)

TAILS = ((0, 0, 135, 345), (180, 0, 256, 375))  # left, right (x0, y0, x1, y1)
REACH = 3
BLUR = 2.0
N8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]


def is_hair(c):
    if c[3] == 0:
        return False
    h, s, v = colorsys.rgb_to_hsv(*(x / 255 for x in c[:3]))
    return s > 0.4 and v > 0.3 and (h < 0.14 or 0.45 < h < 0.6 or h > 0.85)


def tail_region(im):
    w, h = im.size
    px = im.load()
    region = set()
    for x0, y0, x1, y1 in TAILS:
        for y in range(y0, min(y1, h)):
            for x in range(x0, min(x1, w)):
                if not is_hair(px[x, y]):
                    continue
                for dy in range(-REACH, REACH + 1):
                    for dx in range(-REACH, REACH + 1):
                        nx, ny = x + dx, y + dy
                        if x0 <= nx < x1 and y0 <= ny < min(y1, h) \
                                and 0 <= nx < w:
                            region.add((nx, ny))
    return region


def thin_outline(im, region):
    w, h = im.size
    px = im.load()
    changes = []
    for x, y in region:
        if px[x, y] != OUTLINE or is_edge(px, w, h, x, y):
            continue
        if not any(0 <= x + dx < w and 0 <= y + dy < h
                   and px[x + dx, y + dy] == OUTLINE
                   and is_edge(px, w, h, x + dx, y + dy) for dx, dy in N4):
            continue
        hair = [px[x + dx, y + dy] for dx, dy in N8
                if 0 <= x + dx < w and 0 <= y + dy < h
                and is_hair(px[x + dx, y + dy])]
        if len(hair) >= 2:
            changes.append((x, y, Counter(hair).most_common(1)[0][0]))
    for x, y, c in changes:
        px[x, y] = c
    return len(changes)


if __name__ == "__main__":
    src, dst = sys.argv[1:3]
    im = Image.open(src).convert("RGBA")
    region = tail_region(im)
    rem, add = smooth_staircase(im, BLUR, region)
    thin = thin_outline(im, region)
    rep = repair_outline(im, region)
    im.save(dst)
    print(f"drill tails: smoothed (-{rem}/+{add}), thinned {thin} outline "
          f"pixels, redrew {rep} -> {dst}")
