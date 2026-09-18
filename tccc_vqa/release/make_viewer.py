# Builds the offline browser of the package: viewer/index.html and viewer/data.js from the item records. With --frames
# released the viewer reads ../frames/; with --frames all it copies every referenced frame for internal review.
from tccc_vqa.paths import WORK
import argparse, json, os, shutil

POOL = WORK   # source of the non-redistributable stills (review mode only)


def frame_source(fid):
    vid, base = fid.split('__', 1)
    for p in (f'{POOL}/masked_frames/{fid}', f'{POOL}/full_frames/{vid}/{base}'):
        if os.path.exists(p): return p
    return None


HTML = r'''<!doctype html>
<meta charset="utf-8">
<title>TCCC-VQA browser</title>
<style>
 :root{--bg:#fff;--fg:#111;--mut:#666;--line:#ddd;--chip:#f2f2f2;--acc:#0b5}
 *{box-sizing:border-box} body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg);background:var(--bg)}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);padding:10px 16px;z-index:5}
 h1{font-size:16px;margin:0 0 8px} .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
 select,input,button{font:13px inherit;padding:4px 6px;border:1px solid var(--line);border-radius:4px;background:#fff}
 input[type=search]{min-width:260px} .count{color:var(--mut);margin-left:auto}
 main{padding:16px;display:flex;flex-direction:column;gap:16px}
 .item{border:1px solid var(--line);border-radius:6px;padding:12px}
 .hd{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px}
 .id{font-weight:600} .chip{background:var(--chip);border-radius:10px;padding:1px 8px;font-size:12px;color:#333}
 .chip.ref{background:#fde8e8} .chip.gold{background:#fff2cc} .chip.flag{background:#ffe0e0}
 .frames{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
 .fr{position:relative;display:inline-block} .fr img{height:190px;display:block;border-radius:4px;background:#eee}
 .fr .miss{height:190px;width:320px;display:flex;align-items:center;justify-content:center;color:var(--mut);border:1px dashed var(--line);border-radius:4px;font-size:12px;text-align:center;padding:8px}
 .bx{position:absolute;border:2px solid;border-radius:2px}
 .ar{position:absolute;border:2px dashed #d90;border-radius:2px} .ar span{position:absolute;bottom:-15px;right:-2px;background:#d90;color:#fff;font-size:10px;padding:0 3px;border-radius:2px} .bx span{position:absolute;top:-15px;left:-2px;color:#fff;font-size:10px;padding:0 3px;border-radius:2px;white-space:nowrap}
 .cap{font-size:11px;color:var(--mut);margin-top:2px}
 .q{border-top:1px solid #f0f0f0;padding:8px 0} .qt{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
 .qq{font-weight:600;margin:2px 0} .qa{margin:2px 0} .opt{margin:1px 0 1px 14px;color:#333} .opt.gold{color:var(--acc);font-weight:600}
 .prov{font-size:12px;color:var(--mut);margin-top:2px} .prov code{background:var(--chip);padding:0 4px;border-radius:3px}
 .quote{font-size:12px;color:#444;border-left:3px solid var(--line);padding-left:8px;margin-top:3px;white-space:pre-wrap}
 .meta{font-size:12px;color:var(--mut);margin-top:8px;border-top:1px dashed var(--line);padding-top:6px}
 details summary{cursor:pointer;font-size:12px;color:var(--mut)} .hidden{display:none}
</style>
<header>
 <h1>TCCC-VQA <span id="ver" style="font-weight:400;color:#666"></span></h1>
 <div class="row">
  <select id="task"><option value="">task: all</option><option>answerable</option><option>refusal</option></select>
  <select id="concept"><option value="">concept: all</option></select>
  <select id="march"><option value="">MARCH: all</option></select>
  <select id="audit"><option value="">audit: all</option><option>yes</option><option>partial</option></select>
  <select id="prio"><option value="">tier: all</option><option>primary</option><option>review</option></select>
  <select id="pix"><option value="">pixels: all</option><option value="1">released</option><option value="0">by URL only</option></select>
  <select id="gold"><option value="">GOLD: all</option><option value="in">in sample</option><option value="2">both physicians flagged</option></select>
  <select id="det"><option value="">detection: all</option><option value="1">has boxes</option><option value="0">no boxes</option></select>
  <label><input type="checkbox" id="box" checked> boxes</label>
  <label><input type="checkbox" id="ar" checked> answer region</label>
  <input type="search" id="q" placeholder="search id / question / answer / video">
  <span class="count" id="count"></span>
 </div>
</header>
<main id="list"></main>
<script src="data.js"></script>
<script>
const F = window.TCCC_FRAMES_DIR || "../frames/";
const COL = {hand:'#f80', limb_tourniquet:'#e11', junctional_tourniquet:'#c0399b', hemostatic_dressing:'#93c',
  pressure_bandage:'#96f', nasopharyngeal_airway:'#0a8', surgical_airway_device:'#e11', scalpel:'#e63',
  chest_seal:'#096', decompression_needle:'#d02', iv_io_catheter:'#17c', iv_fluid_bag:'#08a', syringe:'#17c',
  hypothermia_wrap:'#786', casualty_card:'#555', marker_pen:'#555', trauma_shears:'#93c'};
document.getElementById('ver').textContent = "version " + DATA.version + " · " + DATA.items.length + " items";
const items = DATA.items, gold = DATA.gold || {};
const uniq = (f) => [...new Set(items.map(f).filter(Boolean))].sort();
for (const c of uniq(i => i.concept)) document.getElementById('concept').add(new Option(c, c));
for (const m of uniq(i => i.march))  document.getElementById('march').add(new Option(m, m));
const esc = s => String(s == null ? "" : s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function frameHTML(fr, showBoxes, ar) {
  const boxes = showBoxes ? (fr.boxes || []).map((b, k) => {
    const c = COL[b.l] || '#e11';
    return `<div class="bx" style="border-color:${c};left:${b.b[0]*100}%;top:${b.b[1]*100}%;width:${(b.b[2]-b.b[0])*100}%;height:${(b.b[3]-b.b[1])*100}%">`
         + `<span style="background:${c};top:${k % 2 ? -15 : -29}px">${esc(b.l)}</span></div>`;
  }).join('') : '';
  const arBox = (ar && fr.has) ? `<div class="ar" style="left:${ar[0]*100}%;top:${ar[1]*100}%;width:${(ar[2]-ar[0])*100}%;height:${(ar[3]-ar[1])*100}%"><span>answer region</span></div>` : '';
  const img = fr.has ? `<img src="${F}${fr.id}" loading="lazy">`
                     : `<div class="miss">not redistributed<br>run scripts/fetch_frames.py</div>`;
  return `<div><div class="fr">${img}${fr.has ? boxes + arBox : ''}</div><div class="cap">${esc(fr.id)} · t=${fr.t}s${(fr.boxes||[]).length ? '' : ' · no detection boxes'}</div></div>`;
}
function qHTML(q) {
  let h = `<div class="q"><div class="qt">${esc(q.type)}${q.facet ? ' · ' + esc(q.facet) : ''}${q.reason_type ? ' · ' + esc(q.reason_type) : ''}</div>`;
  h += `<div class="qq">${esc(q.q)}</div>`;
  if (q.options) h += q.options.map(o => `<div class="opt${o === q.a ? ' gold' : ''}">${esc(o)}</div>`).join('');
  else h += `<div class="qa">${esc(q.an || q.a)}</div>`;
  if (q.rationale) h += `<div class="quote">${esc(q.rationale)}</div>`;
  if (q.p) h += `<div class="prov"><code>${esc(q.p.c)}</code> chars ${q.p.s}–${q.p.e}${q.faithful ? ' · NLI ok' : ''}</div><div class="quote">${esc(q.p.q)}</div>`;
  return h + '</div>';
}
function itemHTML(it, showBoxes, showAR) {
  const g = gold[it.id];
  let h = `<div class="item"><div class="hd"><span class="id">${esc(it.id)}</span>`;
  h += `<span class="chip${it.task === 'refusal' ? ' ref' : ''}">${esc(it.task)}</span>`;
  h += `<span class="chip">${esc(it.concept)}</span><span class="chip">${esc(it.march)}</span>`;
  if (it.safety) h += `<span class="chip">safety-critical</span>`;
  if (it.audit) h += `<span class="chip">audit: ${esc(it.audit)}</span>`;
  if (it.prio) h += `<span class="chip">${esc(it.prio)}</span>`;
  if (it.pix) h += `<span class="chip">pixels released</span>`;
  if (it.task === 'answerable' && !it.nbox) h += `<span class="chip">no detection layer</span>`;
  if (it.task === 'answerable' && !it.ar) h += `<span class="chip">no answer region</span>`;
  if (g) h += `<span class="chip gold">GOLD</span>` + (g.f ? `<span class="chip flag">phys. flagged ×${g.f}</span>` : '');
  h += `</div><div class="frames">${it.frames.map((f, k) => frameHTML(f, showBoxes, (showAR && k === 0) ? it.ar : null)).join('')}</div>`;
  h += it.qs.map(qHTML).join('');
  if (it.mcq) h += `<details><summary>Doctrine-MCQ (easy / hard)</summary>${['easy','hard'].filter(k=>it.mcq[k]).map(k =>
      `<div class="q"><div class="qt">mcq ${k}</div>` + it.mcq[k].map(o => `<div class="opt${o === it.mcq.gold ? ' gold' : ''}">${esc(o)}</div>`).join('') + `</div>`).join('')}</details>`;
  if (it.freegen) h += `<details><summary>Ablation: ungated free-generated answer</summary><div class="quote">${esc(it.freegen)}</div></details>`;
  if (it.see) h += `<details><summary>Independent audit description</summary><div class="quote">${esc(it.see)}</div></details>`;
  h += `<div class="meta">video <a href="${esc(it.url)}" target="_blank">${esc(it.vid)}</a> · ${esc(it.chan)} · segment ${it.seg[0]}–${it.seg[1]}s${it.derived ? ' · frames from ' + esc(it.derived) : ''}</div></div>`;
  return h;
}
function render() {
  const f = {task: task.value, concept: concept.value, march: march.value, audit: audit.value,
             prio: prio.value, pix: pix.value, det: det.value, gold: document.getElementById('gold').value,
             q: q.value.trim().toLowerCase()};
  const showBoxes = box.checked, showAR = document.getElementById('ar').checked;
  const sel = items.filter(it => {
    if (f.task && it.task !== f.task) return false;
    if (f.concept && it.concept !== f.concept) return false;
    if (f.march && it.march !== f.march) return false;
    if (f.audit && it.audit !== f.audit) return false;
    if (f.prio && it.prio !== f.prio) return false;
    if (f.pix && String(it.pix ? 1 : 0) !== f.pix) return false;
    if (f.det && String(it.nbox ? 1 : 0) !== f.det) return false;
    if (f.gold === 'in' && !gold[it.id]) return false;
    if (f.gold === '2' && !(gold[it.id] && gold[it.id].f === 2)) return false;
    if (f.q && !it.blob.includes(f.q)) return false;
    return true;
  });
  document.getElementById('count').textContent = sel.length + ' / ' + items.length;
  document.getElementById('list').innerHTML = sel.slice(0, 300).map(it => itemHTML(it, showBoxes, showAR)).join('') +
    (sel.length > 300 ? `<div class="cap">showing first 300 of ${sel.length}; narrow the filters to see the rest</div>` : '');
}
for (const el of document.querySelectorAll('select,input')) el.addEventListener('input', render);
render();
</script>
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('deposit'); ap.add_argument('out')
    ap.add_argument('--frames', choices=['released', 'all'], default='released')
    a = ap.parse_args()
    D, OUT = a.deposit, a.out
    recs = [json.loads(l) for l in open(f'{D}/data/items.jsonl')]
    mcq = {}
    for kind in ('easy', 'hard'):
        for m in json.load(open(f'{D}/mcq/doctrine_mcq_{kind}.json'))['items']:
            mcq.setdefault(m['id'], {'gold': m['gold']})[kind] = m['options']
    free = {json.loads(l)['item_id']: json.loads(l)['free_gen_answer'] for l in open(f'{D}/eval/ablation_freegen.jsonl')}
    gold_raw = json.load(open(f'{D}/gold/gold_adjudication.json'))['items']
    gold = {k: {'f': v.get('difficulty_flag')} for k, v in gold_raw.items()}
    shipped = set(os.listdir(f'{D}/frames'))

    os.makedirs(OUT, exist_ok=True)
    if a.frames == 'all':
        os.makedirs(f'{OUT}/frames', exist_ok=True)
    items, copied = [], 0
    for r in recs:
        boxes = {}
        for b in (r.get('detection') or []):
            boxes.setdefault(b['frame_id'], []).append({'l': b['label'], 'b': b['bbox_norm']})
        frames = []
        for f in r['frame_refs']:
            fid = f['frame_id']; has = fid in shipped
            if a.frames == 'all':
                src = frame_source(fid)
                if src and not os.path.exists(f'{OUT}/frames/{fid}'):
                    shutil.copy2(src, f'{OUT}/frames/{fid}'); copied += 1
                has = bool(src)
            frames.append({'id': fid, 't': round(f['timestamp_s'], 1), 'has': has, 'boxes': boxes.get(fid, [])})
        qs = []
        for q in r['questions']:
            o = {'type': q['type'], 'q': q['question'], 'a': q['answer']}
            if 'options' in q: o['options'] = q['options']
            if 'answer_normalized' in q: o['an'] = q['answer_normalized']
            if 'provenance' in q:
                p = q['provenance']
                o['p'] = {'c': p['citation_id'], 's': p['char_start'], 'e': p['char_end'], 'q': p.get('source_quote', '')}
            for k in ('facet', 'reason_type', 'faithful', 'rationale'):
                if k in q: o[k] = q[k]
            qs.append(o)
        it = {'id': r['item_id'], 'task': r['task_type'], 'concept': r['concept_id'], 'march': r['march_category'],
              'safety': r.get('safety_critical', False), 'prio': r.get('review_priority'),
              'audit': (r.get('audit') or {}).get('concept_visible'), 'see': (r.get('audit') or {}).get('what_you_see'),
              'pix': r['pixels_released'], 'vid': r['video_id'], 'url': r['source_url'],
              'chan': r.get('source_channel') or r['channel_category'],
              'seg': [r['segment']['t_start_s'], r['segment']['t_end_s']],
              'derived': r.get('derived_from'), 'ar': r.get('answer_region'),
              'nbox': sum(len(f['boxes']) for f in frames), 'frames': frames, 'qs': qs}
        if r['item_id'] in mcq: it['mcq'] = mcq[r['item_id']]
        if r['item_id'] in free: it['freegen'] = free[r['item_id']]
        it['blob'] = ' '.join([r['item_id'], r['concept_id'], r['video_id']] +
                              [q['question'] + ' ' + str(q.get('answer_normalized') or q['answer']) for q in r['questions']]).lower()
        items.append(it)
    version = recs[0].get('release_version', '')
    with open(f'{OUT}/data.js', 'w') as f:
        f.write('const DATA=' + json.dumps({'version': version, 'items': items, 'gold': gold}, ensure_ascii=False) + ';\n')
        if a.frames == 'all': f.write('window.TCCC_FRAMES_DIR="frames/";\n')
    open(f'{OUT}/index.html', 'w').write(HTML)
    n_missing = sum(1 for it in items for fr in it['frames'] if not fr['has'])
    print(f'viewer[{a.frames}] -> {OUT} | {len(items)} items | frames copied {copied} | placeholders {n_missing} '
          f'| data.js {os.path.getsize(f"{OUT}/data.js")/1e6:.1f} MB')


if __name__ == '__main__':
    main()
