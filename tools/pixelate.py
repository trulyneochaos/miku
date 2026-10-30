import sys, numpy as np
from PIL import Image, ImageFilter
from collections import deque
src, H, K, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
im = Image.open(src).convert('RGB')
im = im.filter(ImageFilter.MedianFilter(3))
a = np.asarray(im).astype(int); h, w, _ = a.shape
white = (a.min(2) > 222) & (a.max(2)-a.min(2) < 28)
bg = np.zeros((h,w),bool); q = deque([(y,x) for y in range(h) for x in (0,w-1)] + [(y,x) for x in range(w) for y in (0,h-1)])
while q:
    y,x = q.popleft()
    if 0<=y<h and 0<=x<w and not bg[y,x] and white[y,x]:
        bg[y,x]=True; q.extend(((y+1,x),(y-1,x),(y,x+1),(y,x-1)))
from scipy import ndimage
lbl,n = ndimage.label(white & ~bg)
sz = ndimage.sum(np.ones_like(lbl), lbl, range(1,n+1))
for i,z in enumerate(sz,1):
    if z > 150: bg[lbl==i] = True
ys,xs = np.where(~bg); y0,y1,x0,x1 = ys.min(),ys.max()+1,xs.min(),xs.max()+1
a = a[y0:y1,x0:x1]; fgm = ~bg[y0:y1,x0:x1]; h,w = fgm.shape
W = round(w*H/h)
# palette by k-means on full-res fg pixels
px = a[fgm].astype(float); rng = np.random.default_rng(1)
samp = px[rng.choice(len(px), min(20000,len(px)), replace=False)]
C = samp[rng.choice(len(samp), K, replace=False)]
for _ in range(50):
    l = ((samp[:,None]-C[None])**2).sum(2).argmin(1)
    for k in range(K):
        if (l==k).any(): C[k] = samp[l==k].mean(0)
L = np.full((h,w),-1)
L[fgm] = ((px[:,None]-C[None])**2).sum(2).argmin(1)
lum = C@[0.3,0.59,0.11]; wt = np.where(lum<70, 1.6, 1.0)
# block majority downscale
lab = np.full((H,W),-1)
for Y in range(H):
    for X in range(W):
        blk = L[Y*h//H:max((Y+1)*h//H,Y*h//H+1), X*w//W:max((X+1)*w//W,X*w//W+1)].ravel()
        if (blk>=0).mean() < 0.45: continue
        v,c = np.unique(blk[blk>=0], return_counts=True); c = c*wt[v]; lab[Y,X] = v[c.argmax()]
def nb4(y,x):
    return [(yy,xx) for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)) if 0<=yy<H and 0<=xx<W]
for _ in range(2):
    new = lab.copy()
    for y in range(H):
        for x in range(W):
            if lab[y,x] < 0: continue
            if lab[y,x] not in [lab[p] for p in nb4(y,x)]:
                nb8 = [lab[yy,xx] for yy in (y-1,y,y+1) for xx in (x-1,x,x+1) if 0<=yy<H and 0<=xx<W and (yy,xx)!=(y,x) and lab[yy,xx]>=0]
                if nb8:
                    v,c = np.unique(nb8, return_counts=True)
                    if c.max() >= 3: new[y,x] = v[c.argmax()]
    lab = new
for y in range(H):
    for x in range(W):
        if lab[y,x]>=0 and all(lab[p]<0 for p in nb4(y,x)): lab[y,x]=-1
C = C.round().astype(np.uint8)
dark = int(np.argmin(C.astype(int).sum(1)))
# outline: silhouette edge pixels -> darkest colour
edge = [(y,x) for y in range(H) for x in range(W) if lab[y,x]>=0 and (len(nb4(y,x))<4 or any(lab[p]<0 for p in nb4(y,x)))]
for p in edge: lab[p] = dark
o = np.zeros((H,W,4),np.uint8); m = lab>=0
o[m,:3] = C[lab[m]]; o[m,3]=255
Image.fromarray(o,'RGBA').save(out)
s = max(1, 1024//H)
bgc = Image.new('RGBA',(W*s,H*s),(255,255,255,255))
bgc.alpha_composite(Image.fromarray(o,'RGBA').resize((W*s,H*s),Image.NEAREST))
bgc.convert('RGB').save(out.replace('.png','_preview.png'))
print(W,H,C.tolist())
