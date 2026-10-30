import sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from sklearn.cluster import KMeans

SRC, OUT = sys.argv[1], sys.argv[2]
NCOL = int(sys.argv[3]) if len(sys.argv) > 3 else 40

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


# ---------- 1. background -> transparent (flood fill from the border) ----------
whiteish = (rgb.min(2) > 228) & (rgb.max(2) - rgb.min(2) < 18)
lab, _ = ndi.label(whiteish)
border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
bg = np.isin(lab, border[border > 0])
szw = ndi.sum(whiteish, lab, range(1, lab.max() + 1))
bg |= np.isin(lab, 1 + np.flatnonzero(szw >= 30))
fg = ~bg
# fringe: AA pixels blended toward white that touch the background
for _ in range(1):
    near_bg = ndi.binary_dilation(bg, np.ones((3, 3))) & fg
    fringe = near_bg & (rgb.min(2) > 200) & (rgb.max(2) - rgb.min(2) < 40)
    fg &= ~fringe
fg = ndi.binary_opening(fg, np.ones((2, 2))) | (fg & ndi.binary_erosion(fg, np.ones((2, 1))))
# keep only the main body + anything sizeable
lab, n = ndi.label(fg, np.ones((3, 3)))
sizes = ndi.sum(fg, lab, range(1, n + 1))
fg = np.isin(lab, 1 + np.flatnonzero(sizes >= 12))


# ---------- 2. palette quantization in Lab space ----------
def to_lab(c):
    c = c / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


# light pre-smoothing that respects edges (median) to kill JPEG/AA noise
med = np.stack([ndi.median_filter(rgb[..., i], size=3) for i in range(3)], -1)
src = rgb

L = to_lab(src)
pts = L[fg]
km = KMeans(n_clusters=NCOL, n_init=4, random_state=0).fit(pts[:: max(1, len(pts) // 40000)])
idx = np.full((H, W), -1, int)
idx[fg] = km.predict(pts)
# palette colour = median of the source pixels in each cluster (keeps it vivid)
pal = np.array([np.median(rgb[fg][idx[fg] == k], axis=0) if (idx[fg] == k).any() else [0, 0, 0]
                for k in range(NCOL)])
pal_lab = to_lab(pal)

# merge near-duplicate palette entries
order = np.argsort(pal_lab[:, 0])
remap = np.arange(NCOL)
for i in range(NCOL):
    for j in range(i):
        if remap[j] == j and np.linalg.norm(pal_lab[i] - pal_lab[j]) < 4:
            remap[i] = j
            break
idx[fg] = remap[idx[fg]]


LINE = np.zeros((H, W), bool)
LINE[fg] = pal_lab[idx[fg], 0] < 28

# ---------- 3. de-noise: kill orphan pixels / AA speckle (mode filter) ----------
def mode_pass(idx, strict):
    out = idx.copy()
    counts = {}
    for dy, dx in N8:
        s = shift(idx, dy, dx, -1)
        for k in np.unique(s):
            counts.setdefault(k, np.zeros((H, W), int))
            counts[k] += (s == k)
    keys = np.array(list(counts))
    stack = np.stack([counts[k] for k in keys])
    best = keys[stack.argmax(0)]
    bestn = stack.max(0)
    same = np.zeros((H, W), int)
    for dy, dx in N8:
        same += shift(idx, dy, dx, -2) == idx
    # pixel with <=strict same-coloured neighbours gets the dominant neighbour colour
    change = fg & ~LINE & (same <= strict) & (best >= 0) & (bestn >= 3)
    out[change] = best[change]
    return out


for _ in range(2):
    idx = mode_pass(idx, 0)
# (single-neighbour pass dropped: it ate mouths/eyes)
idx[~fg] = -1

# ---------- 4. outline: darkest palette colour on the silhouette edge ----------
lum = pal_lab[:, 0]
dark = int(np.argmin(np.where(np.bincount(idx[fg], minlength=NCOL) > 0, lum, 999)))
edge = fg & ~ndi.binary_erosion(fg, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), border_value=0)
# only outline where the pixel isn't already dark (avoid doubling)
idx[edge & (lum[np.maximum(idx, 0)] > 30)] = dark


# ---------- 5. dejaggy ----------
def is_(a, dy, dx, k):
    return shift(a, dy, dx, -3) == k


# (a) pixel-perfect: remove L-corner "doubles" in 1-px dark lines
def pixel_perfect(idx, k):
    m = idx == k
    for _ in range(3):
        changed = False
        for (ay, ax), (by, bx) in [((-1, 0), (0, 1)), ((0, 1), (1, 0)), ((1, 0), (0, -1)), ((0, -1), (-1, 0))]:
            a = shift(m, ay, ax, False); b = shift(m, by, bx, False)
            diag = shift(m, ay + by, ax + bx, False)
            opp1 = shift(m, -ay, -ax, False); opp2 = shift(m, -by, -bx, False)
            cand = m & a & b & ~diag & ~opp1 & ~opp2
            if cand.any():
                ys, xs = np.nonzero(cand)
                for y, x in zip(ys, xs):
                    if not m[y, x]:
                        continue
                    # replace with most common non-k neighbour
                    nb = [idx[y + dy, x + dx] for dy, dx in N8
                          if 0 <= y + dy < H and 0 <= x + dx < W and idx[y + dy, x + dx] != k]
                    if nb:
                        idx[y, x] = max(set(nb), key=nb.count)
                        m[y, x] = False
                        changed = True
        if not changed:
            break
    return idx


idx = pixel_perfect(idx, dark)

# (b) silhouette: fill 1-px notches, shave 1-px spurs (on the alpha mask)
for _ in range(2):
    solid = idx >= 0
    n4 = sum(shift(solid, dy, dx, False).astype(int) for dy, dx in N4)
    notch = ~solid & (n4 >= 3)
    spur = solid & (n4 <= 1)
    idx[spur] = -1
    if notch.any():
        idx[notch] = dark

# (c) stair-step cleanup on colour boundaries: a pixel whose 4-neighbours agree
#     on one other colour on 3 sides is a jag -> take that colour
for _ in range(1):
    cnt = {}
    for dy, dx in N4:
        s = shift(idx, dy, dx, -9)
        for k in np.unique(s):
            if k < -1:
                continue
            cnt.setdefault(k, np.zeros((H, W), int))
            cnt[k] += s == k
    keys = np.array(list(cnt)); st = np.stack([cnt[k] for k in keys])
    best = keys[st.argmax(0)]
    diag_same = sum((shift(idx, dy, dx, -9) == idx).astype(int) for dy, dx in [(-1,-1),(-1,1),(1,-1),(1,1)])
    jag = ~LINE & (diag_same == 0) & (st.max(0) >= 3) & (best != idx) & (idx >= 0) & (best >= 0)
    idx[jag] = best[jag]

# re-establish a clean outline after cleanup
solid = idx >= 0
edge = solid & ~ndi.binary_erosion(solid, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), border_value=0)
idx[edge] = dark
idx = pixel_perfect(idx, dark)

# ---------- write ----------
out = np.zeros((H, W, 4), np.uint8)
m = idx >= 0
out[m, :3] = np.clip(pal[idx[m]], 0, 255).astype(np.uint8)
out[m, 3] = 255
Image.fromarray(out, "RGBA").save(OUT)
used = len(np.unique(out[m][:, :3], axis=0))
print("size", W, H, "colours", used)
