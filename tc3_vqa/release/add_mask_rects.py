# Recovers the caption-mask rectangles of each masked frame by diffing it against the full frame, and writes them into
# the release records and meta/mask_rects.json.
from tc3_vqa.paths import WORK
import json, os, numpy as np
from PIL import Image
P=WORK; REL=f'{P}/release'
def rects_for(vid, base):
    fp=f'{P}/full_frames/{vid}/{base}'; mp=f'{P}/masked_frames/{vid}__{base}'
    if not (os.path.exists(fp) and os.path.exists(mp)): return [], os.path.exists(fp)
    a=np.asarray(Image.open(fp).convert('L')).astype(int); b=np.asarray(Image.open(mp).convert('L')).astype(int)
    if a.shape!=b.shape: return [], True
    H,W=a.shape; d=np.abs(a-b)>20
    rows=np.where(d.any(1))[0]
    if len(rows)==0: return [], True
    rects=[]; start=rows[0]; prev=rows[0]
    for r in rows[1:]:
        if r!=prev+1:
            band=d[start:prev+1]; cols=np.where(band.any(0))[0]
            rects.append([round(cols.min()/W,4),round(start/H,4),round((cols.max()+1)/W,4),round((prev+1)/H,4)]); start=r
        prev=r
    band=d[start:prev+1]; cols=np.where(band.any(0))[0]
    rects.append([round(cols.min()/W,4),round(start/H,4),round((cols.max()+1)/W,4),round((prev+1)/H,4)])
    return rects, True
core=f'{REL}/core/tc3_vqa.jsonl'; recs=[json.loads(l) for l in open(core)]
allr={}; n_masked=n_frames=n_missing_full=0
for r in recs:
    for ref in r['frame_refs']:
        fid=ref['frame_id']; base=fid.split('__',1)[1]
        if fid not in allr:
            rects,has_full=rects_for(ref['video_id'],base); allr[fid]=rects
            n_frames+=1; n_masked+=bool(rects); n_missing_full+=(not has_full)
        ref['mask_rects']=allr[fid]
with open(core,'w') as f:
    for r in recs: f.write(json.dumps(r)+'\n')
json.dump(allr,open(f'{REL}/meta/mask_rects.json','w'))
print(f"frames {n_frames} | with mask rects {n_masked} | full frame missing {n_missing_full}")
