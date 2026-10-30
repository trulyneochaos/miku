"""Minimal pixel-art cleanup: keep the artwork, remove only the blur.

1. Transparent background (white connected to the border + enclosed white gaps).
2. Toggle (shock) filter: a pixel that is a blend between two neighbouring
   colours snaps to the nearer one; flat areas and shading are untouched.
3. Gentle palette snap (many colours) so near-identical shades become flats.
4. Dejaggy: only true orphans and 1-px silhouette notches/spurs are fixed.
"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from sklearn.cluster import KMeans

SRC, OUT = sys.argv[1], sys.argv[2]
NCOL = int(sys.argv[3]) if len(sys.argv) > 3 else 96

rgb = np.asarray(Image.open(SRC).convert("RGB")).astype(np.float64)
H, W, _ = rgb.shape
N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
N4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]


def shift(a, dy, dx, fill):
    out = np.full_like(a, fill)
    ys, yd = (slice(dy, None), slice(0, H - dy)) if dy >= 0 else (slice(0, H + dy), slice(-dy, None))
    xs, xd = (slice(dx, None), slice(0, W - dx)) if dx >= 0 else (slice(0, W + dx), slice(-dx, None))
    out[yd, xd] = a[ys, xs]
    return out


# ---------- 1. background ----------
white = (rgb.min(2) > 238) & (rgb.max(2) - rgb.min(2) < 12)
lab, n = ndi.label(white)
border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
sz = ndi.sum(white, lab, range(1, n + 1))
bg = np.isin(lab, border[border > 0]) | np.isin(lab, 1 + np.flatnonzero(sz >= 40))
# one ring of white-blended halo pixels next to the background
halo = ndi.binary_dilation(bg) & ~bg & (rgb.min(2) > 215) & (rgb.max(2) - rgb.min(2) < 30)
bg |= halo
fg = ~bg

img = rgb

# ---------- 3. gentle palette snap ----------
pts = img[fg]
km = KMeans(n_clusters=NCOL, n_init=2, random_state=0).fit(pts[:: max(1, len(pts) // 30000)])
idx = np.full((H, W), -1, int)
idx[fg] = km.predict(pts)
pal = km.cluster_centers_

# ---------- 4. dejaggy ----------
def nbcount(idx, k_arr, offs):
    return sum((shift(idx, dy, dx, -9) == k_arr).astype(int) for dy, dx in offs)

for _ in range(2):
    same8 = nbcount(idx, idx, N8)
    orphan = fg & (same8 == 0)
    # orphan -> nearest-coloured 4-neighbour (keeps tiny highlights only if they
    # have any same-coloured neighbour)
    if orphan.any():
        best = np.full((H, W), -1); bestd = np.full((H, W), np.inf)
        for dy, dx in N4:
            s = shift(idx, dy, dx, -1)
            ok = s >= 0
            dd = np.where(ok, np.linalg.norm(pal[np.maximum(s, 0)] - pal[np.maximum(idx, 0)], axis=-1), np.inf)
            better = ok & (dd < bestd)
            best[better] = s[better]; bestd[better] = dd[better]
        m = orphan & (best >= 0) & (bestd < 45)
        idx[m] = best[m]

# silhouette: 1-px notches filled, 1-px spurs removed
for _ in range(2):
    solid = idx >= 0
    n4 = sum(shift(solid, dy, dx, False).astype(int) for dy, dx in N4)
    spur = solid & (n4 <= 1)
    idx[spur] = -1
    notch = ~solid & (n4 >= 3)
    for y, x in zip(*np.nonzero(notch)):
        c = [idx[y + dy, x + dx] for dy, dx in N4 if 0 <= y + dy < H and 0 <= x + dx < W and idx[y + dy, x + dx] >= 0]
        idx[y, x] = max(set(c), key=c.count)

out = np.zeros((H, W, 4), np.uint8)
m = idx >= 0
out[m, :3] = np.clip(pal[idx[m]], 0, 255).round().astype(np.uint8)
out[m, 3] = 255
Image.fromarray(out, "RGBA").save(OUT)
print("colours", len(np.unique(out[m][:, :3], axis=0)))
