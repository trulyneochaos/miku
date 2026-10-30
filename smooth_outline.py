#!/usr/bin/env python3
"""Smooth staircase edges and repair the outline of a pixel-art sprite.

1. Staircase: blur the alpha mask and re-threshold it, which evens out
   uneven step lengths along the silhouette. Guards keep detail intact:
   - thin pixels (<=2 opaque 4-neighbours: hair strands, line tips) are
     never removed
   - only background reachable from the canvas edge is filled, so enclosed
     holes are left alone, and a fill needs >=3 opaque 4-neighbours or an
     L-shaped pair (never a straight 1px channel)
2. Outline: every opaque pixel touching transparency is set to the
   outline colour, closing gaps and re-covering edge pixels exposed or
   added by the smoothing.

usage: smooth_outline.py input.png output.png [blur_radius]
"""
import sys
from collections import deque
from PIL import Image, ImageFilter

OUTLINE = (0, 0, 0, 255)
N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def solid(px, w, h, x, y):
    return 0 <= x < w and 0 <= y < h and px[x, y][3] > 0


def is_edge(px, w, h, x, y):
    return solid(px, w, h, x, y) and any(
        not solid(px, w, h, x + dx, y + dy) for dx, dy in N4)


def outside(px, w, h):
    """Transparent pixels 4-connected to the canvas border."""
    seen = set()
    q = deque((x, y) for x in range(w) for y in (0, h - 1)
              if px[x, y][3] == 0)
    q.extend((x, y) for y in range(h) for x in (0, w - 1)
             if px[x, y][3] == 0)
    seen.update(q)
    while q:
        x, y = q.popleft()
        for dx, dy in N4:
            n = (x + dx, y + dy)
            if (0 <= n[0] < w and 0 <= n[1] < h and n not in seen
                    and px[n][3] == 0):
                seen.add(n)
                q.append(n)
    return seen


def smooth_staircase(im, radius, region=None):
    w, h = im.size
    px = im.load()
    mask = im.getchannel("A").point(lambda a: 255 if a else 0)
    smooth = mask.filter(ImageFilter.GaussianBlur(radius)).load()
    bg = outside(px, w, h)
    removed = added = 0
    changes = []
    for y in range(h):
        for x in range(w):
            if region is not None and (x, y) not in region:
                continue
            on = smooth[x, y] >= 128
            nb = [(dx, dy) for dx, dy in N4
                  if solid(px, w, h, x + dx, y + dy)]
            n = len(nb)
            corner = n == 2 and nb[0][0] + nb[1][0] != 0 \
                and nb[0][1] + nb[1][1] != 0
            if px[x, y][3] > 0 and not on and n >= 3:
                changes.append((x, y, (0, 0, 0, 0)))
                removed += 1
            elif px[x, y][3] == 0 and on and (n >= 3 or corner) and (x, y) in bg:
                changes.append((x, y, OUTLINE))
                added += 1
    for x, y, c in changes:
        px[x, y] = c
    return removed, added


def repair_outline(im, region=None):
    w, h = im.size
    px = im.load()
    todo = [(x, y) for y in range(h) for x in range(w)
            if is_edge(px, w, h, x, y) and px[x, y] != OUTLINE
            and (region is None or (x, y) in region)]
    for x, y in todo:
        px[x, y] = OUTLINE
    return len(todo)


if __name__ == "__main__":
    src, dst = sys.argv[1:3]
    radius = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    im = Image.open(src).convert("RGBA")
    rem, add = smooth_staircase(im, radius)
    rep = repair_outline(im)
    im.save(dst)
    print(f"smoothed edge (-{rem}/+{add} pixels), "
          f"repaired {rep} outline pixels -> {dst}")
