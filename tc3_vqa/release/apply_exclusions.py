# Builds the redistribution-safe release: drops items whose source video is unavailable and items the recognition
# consensus did not confirm, replaces pixel paths by video id and timestamp, keeps images only for permitted sources.
from tc3_vqa.paths import EXPERIMENTS, WORK
import json, csv, os, re, shutil, sys
from collections import defaultdict, Counter

P=WORK
SRC=f'{P}/release_full'
OUT=f'{P}/release'
AUDIT=f'{P}/source_video_license_audit.csv'
YTMETA=f'{P}/source_videos_ytmeta.csv'   # channel/uploader/licence from yt-dlp (fetch_source_meta.py)
GOV_RE=r'CoTCCC|Committee.on.TCCC|Joint Trauma|\\bJTS\\b|U\\.?S\\.? ?Army|Army Medic|MEDCoE|Defense Health|\\bDHA\\b|Department of Defense|\\bDoD\\b|U\\.?S\\.? Navy|Marine Corps|Air Force|National Guard|Military Health|Uniformed Services|AMEDD|Medical Center of Excellence|DVIDS|Public Affairs'
WINDOWS=f'{P}/pass0_windows.json'   # optional: exact PASS0 windows {video_id:[[si,wi,ws,we],...]}
VERSION='1.0'
GATE=EXPERIMENTS + '/consensus_perception/gate_3of6.json'   # consensus_analyze.py / gate list
DROP_CATS={'removed','private'}
LOGIN_CATS={'age_restricted_standard'}
PIXEL_CATS={'creative_commons'}
MARCH={'tourniquet_application':'M','wound_packing':'M','junctional_hemorrhage':'M','tourniquet_conversion':'M',
       'nasopharyngeal_airway':'A','surgical_airway_cric':'A','chest_seal':'R','needle_decompression':'R',
       'iv_io_access':'C','txa_administration':'C','hypothermia_prevention':'H','tccc_documentation':'process'}
FN=re.compile(r'^(?:masked_frames__)?(?P<vid>[A-Za-z0-9_-]+?)__shot(?P<si>\d{4})_w(?P<wi>\d{2})_c(?P<ci>\d{2})_t(?P<ms>\d{8})\.jpg$')

aud={r['video_id']:r for r in csv.DictReader(open(AUDIT))}
ytm={r['video_id']:r for r in csv.DictReader(open(YTMETA))} if os.path.exists(YTMETA) else {}
def channel_of(v):
    m=ytm.get(v,{}); return m.get('channel') or m.get('uploader') or aud.get(v,{}).get('uploader','') or ''
def channel_cat(v):
    m=ytm.get(v,{}); ch=channel_of(v)
    if aud.get(v,{}).get('license_category')=='creative_commons' or 'creative' in (m.get('license') or '').lower(): return 'creative_commons'
    if re.search(GOV_RE,ch,re.I): return 'us_government_public_domain'
    return 'standard' if ch else 'standard_unknown_channel'
PIX_CATS={'creative_commons','us_government_public_domain'}
items=[json.loads(l) for l in open(f'{SRC}/core/tc3_vqa.jsonl')]
windows=json.load(open(WINDOWS)) if os.path.exists(WINDOWS) else None

def parse_frame(path):
    m=FN.match(os.path.basename(path))
    if not m: raise SystemExit(f'unparseable frame name: {path}')
    d=m.groupdict(); return d['vid'], int(d['si']), int(d['wi']), int(d['ci']), int(d['ms'])/1000.0
def vid_of(it):
    return it.get('video_id') or parse_frame(it['frame_paths'][0])[0]

kept, dropped = [], []
for it in items:
    v=vid_of(it); cat=aud.get(v,{}).get('license_category','unknown')
    (dropped if cat in DROP_CATS else kept).append((it,v,cat))
# consensus gate: drop answerable items without >= 3/6 agreement and the refusals derived from them
gate=json.load(open(GATE)); gate_drop=set(gate['drop'])
gated=[(it,v,c) for it,v,c in kept if it['item_id'] in gate_drop or (it['task_type']!='answerable' and it.get('src_item') in gate_drop)]
gated_ids={it['item_id'] for it,_,_ in gated}
kept=[(it,v,c) for it,v,c in kept if it['item_id'] not in gated_ids]

if os.path.exists(OUT): shutil.rmtree(OUT)
for d in ['core','segments','meta','gold','mcq','eval','frames','scripts']: os.makedirs(f'{OUT}/{d}',exist_ok=True)

seg_index=defaultdict(lambda:{'item_ids':[],'ts':[]})
records=[]; n_pix=0
for it,v,cat in kept:
    rec=dict(it)
    rec['video_id']=v
    rec['source_url']=f'https://www.youtube.com/watch?v={v}'
    rec['source_license']=cat
    rec['requires_login']=cat in LOGIN_CATS
    rec['source_channel']=channel_of(v)
    rec['channel_category']=channel_cat(v)
    rec['pixels_released']=channel_cat(v) in PIX_CATS
    refs=[]; ts=[]
    for fp in it['frame_paths']:
        fv,si,wi,ci,t=parse_frame(fp); ts.append(t)
        base=os.path.basename(fp); masked_in_release=base.startswith('masked_frames__')
        fid=base[len('masked_frames__'):] if masked_in_release else base
        ref={'frame_id':fid,'video_id':fv,'timestamp_s':round(t,3),'shot_idx':si,'window_idx':wi,'candidate_idx':ci,'masked_in_release':masked_in_release}
        if channel_cat(v) in PIX_CATS:
            src=f'{SRC}/{fp}'
            if os.path.exists(src):
                shutil.copy2(src,f'{OUT}/frames/{fid}'); ref['local_path']=f'frames/{fid}'; n_pix+=1
        refs.append(ref)
    rec['frame_refs']=refs
    rec.pop('frame_paths',None)
    si,wi=refs[0]['shot_idx'],refs[0]['window_idx']
    seg=None
    if windows and v in windows:
        for a,b,ws,we in windows[v]:
            if a==si and b==wi: seg={'shot_idx':si,'window_idx':wi,'t_start_s':round(ws,3),'t_end_s':round(we,3),'source':'pass0_window'}; break
    if seg is None:
        seg={'shot_idx':si,'window_idx':wi,'t_start_s':round(min(ts),3),'t_end_s':round(max(ts),3),'source':'frame_bounds'}
    rec['segment']=seg
    concept=it.get('matched_concept_id')
    rec['march_category']=MARCH.get(concept) if concept else None
    rec['release_version']=VERSION
    key=(v,si,wi); seg_index[key]['item_ids'].append(it['item_id']); seg_index[key]['ts']+=ts
    records.append(rec)

with open(f'{OUT}/core/tc3_vqa.jsonl','w') as f:
    for r in records: f.write(json.dumps(r)+'\n')

# segments table
with open(f'{OUT}/segments/segments.jsonl','w') as f:
    for (v,si,wi),d in sorted(seg_index.items()):
        seg={'segment_id':f'{v}__shot{si:04d}_w{wi:02d}','video_id':v,'source_url':f'https://www.youtube.com/watch?v={v}',
             'source_license':aud[v]['license_category'],'shot_idx':si,'window_idx':wi,
             't_start_s':round(min(d['ts']),3),'t_end_s':round(max(d['ts']),3),'source':'frame_bounds','item_ids':sorted(d['item_ids'])}
        if windows and v in windows:
            for a,b,ws,we in windows[v]:
                if a==si and b==wi: seg.update({'t_start_s':round(ws,3),'t_end_s':round(we,3),'source':'pass0_window'}); break
        f.write(json.dumps(seg)+'\n')

# source video manifest
kept_ids={it['item_id'] for it,_,_ in kept}
per_vid=Counter(v for _,v,_ in kept)
with open(f'{OUT}/meta/source_videos.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['video_id','source_url','license_category','channel','channel_category','requires_login','pixels_released','n_items','title'])
    for v,n in sorted(per_vid.items()):
        c=aud[v]['license_category']; cc=channel_cat(v); w.writerow([v,f'https://www.youtube.com/watch?v={v}',c,channel_of(v),cc,c in LOGIN_CATS,cc in PIX_CATS,n,(ytm.get(v,{}).get('title') or aud[v].get('title',''))])
RIGHTS='source video removed or private at audit time; rights to redistribute stills do not transfer'
CONSENSUS=('recognition not confirmed by the six-model consensus gate (fewer than 3 of 6 independent voters named the '
           'released concept), or a refusal item derived from such an item')
json.dump({'reasons':{'source_rights':RIGHTS,'consensus_gate':CONSENSUS},
           'items':[{'item_id':it['item_id'],'video_id':v,'license_category':c,'task_type':it['task_type'],'reason':'source_rights'} for it,v,c in dropped]
                  +[{'item_id':it['item_id'],'video_id':v,'license_category':c,'task_type':it['task_type'],'reason':'consensus_gate',
                     **({'derived_from':it.get('src_item')} if it['task_type']!='answerable' else {'consensus_concept':gate['consensus'].get(it['item_id'])})} for it,v,c in gated]},
          open(f'{OUT}/meta/dropped_items.json','w'),indent=1)

# gold (filter to kept)
g=json.load(open(f'{SRC}/gold/gold_adjudication.json'))
g_items={k:v for k,v in g['items'].items() if k in kept_ids}
g_drop=sorted(set(g['items'])-set(g_items))
g.update({'items':g_items,'n_items':len(g_items),'dropped_item_ids':g_drop,
          'difficulty_flag_counts':{str(k):sum(1 for v in g_items.values() if v['difficulty_flag']==k) for k in (0,1,2)},
          'release_version':VERSION})
json.dump(g,open(f'{OUT}/gold/gold_adjudication.json','w'),indent=1)

# mcq / eval (filter items lists by id; jsonl by item_id)
def filt_json(src,dst):
    d=json.load(open(src)); 
    if isinstance(d,dict) and 'items' in d: d['items']=[x for x in d['items'] if x['id'] in kept_ids]
    elif isinstance(d,list): d=[x for x in d if x.get('id') in kept_ids]
    json.dump(d,open(dst,'w'))
for fn in ['doctrine_mcq_easy.json','doctrine_mcq_hard.json']: filt_json(f'{SRC}/mcq/{fn}',f'{OUT}/mcq/{fn}')
shutil.copy2(f'{SRC}/mcq/build_doctrine_mcq.py',f'{OUT}/mcq/')
for fn in ['doctrine_open_input.json','how_input.json','recognition_input.json','refusal_input.json']: filt_json(f'{SRC}/eval/{fn}',f'{OUT}/eval/{fn}')
with open(f'{OUT}/eval/ablation_freegen.jsonl','w') as f:
    for l in open(f'{SRC}/eval/ablation_freegen.jsonl'):
        if json.loads(l)['item_id'] in kept_ids: f.write(l)
shutil.copytree(f'{SRC}/eval/scripts',f'{OUT}/eval/scripts')

# strip local frame paths from eval/mcq 'frames' fields (they point at pixels) -> keep frame ids only
for fn in ['mcq/doctrine_mcq_easy.json','mcq/doctrine_mcq_hard.json','eval/doctrine_open_input.json','eval/how_input.json','eval/recognition_input.json','eval/refusal_input.json']:
    p=f'{OUT}/{fn}'; d=json.load(open(p)); lst=d['items'] if isinstance(d,dict) else d
    for x in lst:
        if 'frames' in x: x['frame_ids']=[os.path.basename(z) for z in x['frames']]; del x['frames']
    json.dump(d,open(p,'w'))

na=sum(1 for r in records if r['task_type']=='answerable'); nr=len(records)-na
print(f"kept {len(records)} ({na} answerable + {nr} refusal) | dropped rights {len(dropped)} + consensus gate {len(gated)} | videos {len(per_vid)} | pixels shipped {n_pix} | gold {len(g_items)} (dropped {len(g_drop)}) | segments {len(seg_index)} | windows file: {'YES' if windows else 'no -> frame_bounds'}")
