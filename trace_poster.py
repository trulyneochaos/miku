import cv2, numpy as np, vtracer, re, sys
img=cv2.imread(sys.argv[1] if len(sys.argv)>1 else 'src.jpg')
# Curated poster palette (BGR from hex)
pal_hex=['f1e3d1','b05a57','515b67','52555b','626973','8cafa9','658185','7b98a2','88898a','ecd2c0','c19e95','b5b0a9','3e4450']
pal=np.array([[int(h[i:i+2],16) for i in (4,2,0)] for h in pal_hex],np.uint8)
sm=cv2.pyrMeanShiftFiltering(img,6,18)
lab=cv2.cvtColor(sm,cv2.COLOR_BGR2LAB).reshape(-1,1,3).astype(np.float32)
pl=cv2.cvtColor(pal.reshape(-1,1,3),cv2.COLOR_BGR2LAB).reshape(1,-1,3).astype(np.float32)
lbl=np.argmin(((lab-pl)**2).sum(-1),1).reshape(img.shape[:2]).astype(np.uint8)
# majority (mode) filter to kill JPEG speckle, keep edges crisp
K=len(pal)
for _ in range(2):
    votes=np.stack([cv2.blur((lbl==k).astype(np.float32),(5,5)) for k in range(K)])
    lbl=votes.argmax(0).astype(np.uint8)
# upscale labels 2x with smooth boundaries for nicer curves
H,W=lbl.shape
votes=np.stack([cv2.resize(cv2.GaussianBlur((lbl==k).astype(np.float32),(0,0),1.0),(W*2,H*2),interpolation=cv2.INTER_CUBIC) for k in range(K)])
lbl2=votes.argmax(0)
q=pal[lbl2]
cv2.imwrite('quant.png',q)
vtracer.convert_image_to_svg_py('quant.png','raw.svg',colormode='color',hierarchical='stacked',mode='spline',
    filter_speckle=24,color_precision=8,layer_difference=4,corner_threshold=60,length_threshold=4.0,max_iterations=10,splice_threshold=45,path_precision=1)
s=open('raw.svg').read()
# snap colors to palette & rescale to original pixel size
def snap(m):
    c=m.group(1); r,g,b=int(c[0:2],16),int(c[2:4],16),int(c[4:6],16)
    d=[(int(h[0:2],16)-r)**2+(int(h[2:4],16)-g)**2+(int(h[4:6],16)-b)**2 for h in pal_hex]
    return 'fill="#%s"'%pal_hex[int(np.argmin(d))]
s=re.sub(r'fill="#([0-9A-Fa-f]{6})"',snap,s)
body=s[s.index('>',s.index('<svg'))+1:s.rindex('</svg>')]
out=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W*2} {H*2}" width="{W}" height="{H}">\n<title>Poster trace</title>\n{body}</svg>\n'
open('poster.svg','w').write(out)
print(len(out), out.count('<path'))
