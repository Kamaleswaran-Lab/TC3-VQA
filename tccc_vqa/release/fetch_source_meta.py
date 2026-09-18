# Fetches channel, licence and availability of every source video with yt-dlp, without downloading, and writes the
# source manifest used to decide which frames may be redistributed.
from tccc_vqa.paths import RELEASE_PARENT
import csv, json, sys, yt_dlp
REL=RELEASE_PARENT
vids=[r['video_id'] for r in csv.DictReader(open(f'{REL}/meta/source_videos.csv'))]
opts={'quiet':True,'skip_download':True,'no_warnings':True,'ignoreerrors':True}
rows=[]
with yt_dlp.YoutubeDL(opts) as y:
    for i,v in enumerate(vids,1):
        try:
            info=y.extract_info(f'https://www.youtube.com/watch?v={v}',download=False) or {}
            rows.append({'video_id':v,'available':bool(info),'uploader':info.get('uploader') or '','channel':info.get('channel') or '',
                         'channel_id':info.get('channel_id') or '','license':info.get('license') or '','age_limit':info.get('age_limit'),
                         'title':info.get('title') or '','upload_date':info.get('upload_date') or '','duration':info.get('duration')})
            print(f'[{i}/{len(vids)}] {v} | {info.get("channel")} | lic={info.get("license")}',flush=True)
        except Exception as e:
            rows.append({'video_id':v,'available':False,'uploader':'','channel':'','channel_id':'','license':'','age_limit':None,'title':'','upload_date':'','duration':None})
            print(f'[{i}/{len(vids)}] {v} ERR {e}',flush=True)
with open(f'{REL}/meta/source_videos_ytmeta.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('DONE',len(rows))
