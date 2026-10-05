# Writes croissant.json, the Croissant 1.0 metadata of the package, with a sha256 per file.
from tc3_vqa.paths import RELEASE
import json, os, hashlib
DEP=RELEASE; os.chdir(DEP)
ENC={'jsonl':'application/jsonlines','json':'application/json','csv':'text/csv','py':'text/x-python','md':'text/markdown','cff':'text/plain','txt':'text/plain','html':'text/html','js':'text/javascript'}
recs=[json.loads(l) for l in open('data/items.jsonl')]
N=len(recs); NA=sum(r['task_type']=='answerable' for r in recs); NR=N-NA
files=[]
for root,_,fs in os.walk('.'):
    for f in fs:
        p=os.path.join(root,f)[2:]; ext=p.rsplit('.',1)[-1]
        if p.startswith('frames/') or ext not in ENC or p in ('croissant.json','SHA256SUMS'): continue
        files.append(p)
dist=lambda p:{"@type":"cr:FileObject","@id":p,"name":p,"contentUrl":p,"encodingFormat":ENC[p.rsplit('.',1)[-1]],"sha256":hashlib.sha256(open(p,"rb").read()).hexdigest()}
fields=[("item_id","Text","stable item identifier"),("task_type","Text","answerable or refusal"),("release_version","Text","release tag"),
        ("concept_id","Text","one of 12 TCCC concepts (scene concept for refusal items)"),("march_category","Text","M/A/R/C/H/process"),("safety_critical","Boolean","RWHR stakes flag (answerable)"),
        ("review_priority","Text","primary / review (answerable)"),
        ("video_id","Text","YouTube video id"),("source_url","URL","public source video"),("source_license","Text","license category of the source"),
        ("source_channel","Text","uploading channel"),("channel_category","Text","us_government_public_domain / creative_commons / standard / standard_unknown_channel"),
        ("requires_login","Boolean","age-restricted source"),("pixels_released","Boolean","frame pixels included in frames/"),
        ("segment","Text","shot/window index and t_start_s/t_end_s of the source segment"),("frame_refs","Text","frame id, timestamp_s, mask_rects, optional local_path"),
        ("questions","Text","recognition_mcq / doctrine_scene / reasoning / how / refusal with verbatim answers and char-offset provenance"),
        ("detection","Text","reviewed equipment boxes (17 classes, bbox_norm), same as data/detection_coco.json (answerable)"),("answer_region","Text","normalized evidence box (answerable)"),
        ("observed_anatomy","Text","observed body region and confidence (answerable)"),("audit","Text","generator-blind independent audit of the frames (answerable)"),
        ("auto_checks","Text","automatic verification flags (answerable)"),("refusal_kind","Text","refusal derivation (refusal)"),("refusal_reason","Text","category of missing evidence (refusal)"),
        ("derived_from","Text","answerable item whose frames are reused (refusal)")]
cfields=[("citation_id","Text","chunk identifier cited by provenance.citation_id"),("source_id","Text","source document"),("section_path","Text","heading path within the document"),("page_start","Integer","first page"),("page_end","Integer","last page"),("text","Text","chunk text; cited answers are text[char_start:char_end]")]
rs={"@type":"cr:RecordSet","@id":"items","name":"items","description":f"One record per TC3-VQA item ({N}). Field dictionary: meta/FIELDS.md.",
    "field":[{"@type":"cr:Field","@id":f"items/{n}","name":n,"description":d,"dataType":f"sc:{t}","source":{"fileObject":{"@id":"data/items.jsonl"},"extract":{"column":n}}} for n,t,d in fields]}
cro={"@context":{"@language":"en","@vocab":"https://schema.org/","cr":"http://mlcommons.org/croissant/","sc":"https://schema.org/","dct":"http://purl.org/dc/terms/","citeAs":"cr:citeAs","conformsTo":"dct:conformsTo","data":{"@id":"cr:data","@type":"@json"},"dataType":{"@id":"cr:dataType","@type":"@vocab"},"extract":"cr:extract","field":"cr:field","fileObject":"cr:fileObject","recordSet":"cr:recordSet","source":"cr:source","column":"cr:column"},
     "@type":"sc:Dataset","conformsTo":"http://mlcommons.org/croissant/1.0","name":"TC3-VQA","version":"1.0",
     "description":f"Doctrine-grounded, refusal-aware, image-grounded question answering benchmark for Tactical Combat Casualty Care: {N} items ({NA} answerable, {NR} in-domain refusal) over 11 TCCC concepts. Answerable items carry a recognition question, a doctrine question, a clinical-reasoning question and a procedural question; doctrine answers are verbatim spans of public TCCC documents cited by character offset, and the cited chunks are included. Frames are identified by source video and timestamp; image files are included only for public-domain and Creative-Commons sources, and the others are regenerated with scripts/fetch_frames.py.",
     "license":"https://creativecommons.org/licenses/by/4.0/","url":"https://doi.org/PLACEHOLDER","citeAs":"TO_FILL_AT_DEPOSIT",
     "keywords":["Tactical Combat Casualty Care","medical VQA","vision-language","refusal","doctrine grounding","benchmark"],
     "distribution":[dist(p) for p in sorted(files)],"recordSet":[rs,{"@type":"cr:RecordSet","@id":"doctrine_chunks","name":"doctrine_chunks","description":"Doctrine corpus chunks cited by the answers.","field":[{"@type":"cr:Field","@id":f"doctrine_chunks/{n}","name":n,"description":d,"dataType":f"sc:{t}","source":{"fileObject":{"@id":"data/doctrine_chunks.jsonl"},"extract":{"column":n}}} for n,t,d in cfields]}]}
json.dump(cro,open('croissant.json','w'),indent=1); print("croissant:",len(files),"file objects")
