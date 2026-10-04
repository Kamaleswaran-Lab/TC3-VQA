# Builds the public package in one run: exclusions, mask rectangles, raters' notes, detection file, public schema,
# frame features, Croissant metadata and checksums. The authored README, LICENSE and CHANGELOG are kept.
from tc3_vqa.paths import RELEASE_PARENT, WORK
import subprocess, shutil, os, json, sys
HERE=os.path.dirname(os.path.abspath(__file__)); EV=os.path.join(HERE,'..','eval'); P=WORK
REL=f'{P}/release'; ROOT=RELEASE_PARENT; DEP=RELEASE; PY=sys.executable
KEEP_DOCS=['README.md','LICENSE.md','CHANGELOG.md']   # authored docs live in the deposit; preserved across rebuilds
saved={d:open(f'{DEP}/{d}').read() for d in KEEP_DOCS if os.path.exists(f'{DEP}/{d}')}
assert set(saved)==set(KEEP_DOCS), f'authored docs missing from the deposit: {set(KEEP_DOCS)-set(saved)}; restore them before rebuilding'
subprocess.run([PY,f'{HERE}/apply_exclusions.py'],check=True)
subprocess.run([PY,f'{HERE}/add_mask_rects.py'],check=True)
subprocess.run([PY,os.path.join(HERE,'..','adjudication','gold_notes.py')],check=True)   # raters' notes into the adjudication file
# detection COCO filtered to shipped frame ids
recs=[json.loads(l) for l in open(f'{REL}/core/tc3_vqa.jsonl')]
kept_f={f['frame_id'] for r in recs for f in r['frame_refs']}
coco=json.load(open(f'{P}/release_full/core/detection_coco.json'))
def key(im): p=im['file_name'].split('/'); return (f"{p[-2]}__{p[-1]}" if len(p)>=2 else p[-1]).replace('masked_frames__','')
imgs=[dict(im,file_name=key(im)) for im in coco['images'] if key(im) in kept_f]; ids={im['id'] for im in imgs}
json.dump({'images':imgs,'annotations':[a for a in coco['annotations'] if a['image_id'] in ids],'categories':coco['categories']},open(f'{REL}/core/detection_coco.json','w'))
os.makedirs(f'{REL}/scripts',exist_ok=True); shutil.copy2(f'{HERE}/fetch_frames.py',f'{REL}/scripts/')
shutil.copy2(f'{P}/source_videos_ytmeta.csv',f'{REL}/meta/source_videos_ytmeta.csv')
# public deposit
subprocess.run([PY,f'{HERE}/clean_release.py',REL,DEP],check=True)
for d,txt in saved.items(): open(f'{DEP}/{d}','w').write(txt)
# internal review viewer only (all frames); the package ships no viewer
subprocess.run([PY,f'{HERE}/make_viewer.py',DEP,f'{ROOT}/review','--frames','all'],check=True)
ART=f'{ROOT}/artifacts'                                   # frame features, computed separately on a GPU node
if os.path.exists(f'{ART}/frame_features.npz'):
    for f in ('frame_features.npz','frame_features.json'): shutil.copy2(f'{ART}/{f}',f'{DEP}/data/{f}')
    print('frame features: copied from artifacts')
else:
    print('frame features: missing, run tc3vlm/qa/frame_features.py')
subprocess.run([PY,f'{HERE}/make_croissant.py'],check=True)
import hashlib
with open(f'{DEP}/SHA256SUMS','w') as f:
    for root,_,fs in os.walk(DEP):
        for name in sorted(fs):
            p=os.path.join(root,name); rel=os.path.relpath(p,DEP)
            if rel=='SHA256SUMS': continue
            f.write(f"{hashlib.sha256(open(p,'rb').read()).hexdigest()}  {rel}\n")
print(f"deposit synced: {DEP} | records {len(recs)} | pixels shipped {len(os.listdir(f'{DEP}/frames'))} | COCO {len(imgs)} imgs / {sum(1 for a in coco['annotations'] if a['image_id'] in ids)} boxes")
