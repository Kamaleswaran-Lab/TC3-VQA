# Review of equipment-detection boxes

Step: Annotation: every refined box, drawn and numbered on its frame; applied by `tc3_vqa/annotation/apply_box_review.py`, missed objects re-detected by `recall_detect.py`.

Input: A manifest slice of `{frame_id, file, boxes: [{idx, label, verify}]}`, the frames with boxes drawn, and the ontology.

Output: JSON `{corrections: [{frame_id, boxes: [{idx, verdict, new_label, note}], missed: [{label, note}]}]}` written to the batch output file.

The prompt follows as it was run; the batch instructions inside it are handled by scripts/run_prompt_batch.py.

```
You are an expert visual annotator auditing auto-generated equipment DETECTION boxes on Tactical Combat Casualty Care (TCCC) video frames. Be strict and judge ONLY what is actually inside each drawn box, not what the procedure implies.

STEP 1 — get your batch metadata (frames {first}..{last}):
  python3 -c "import json;d=json.load(open('{batch_file}'));print(json.dumps(d[{first}:{last}]))"
Each entry = {frame_id, file, boxes:[{idx,label,verify}]}.

STEP 2 — for each entry, Read the image file at {frames_dir}/<file>. Boxes are drawn and numbered as "#idx label".

STEP 3 — judge EVERY numbered box:
 - "keep": the box correctly localizes a real object of that label.
 - "relabel" + new_label: a real object is boxed but the label is WRONG (give the correct label id).
 - "delete": no such object / false positive / box not on the object / box is on a burned-in caption or logo (captions/logos are NOT objects).
Also list "missed": clearly-visible equipment from the ontology that has NO box (recall). Approximate note only; do not invent coordinates.

ONTOLOGY (valid label ids):
{ontology}

STEP 4 — Write your result as JSON to {out_dir}/batch_{first}.json with EXACTLY this shape:
{"corrections":[{"frame_id":"<id>","boxes":[{"idx":<int>,"verdict":"keep|relabel|delete","new_label":"<id if relabel else empty>","note":"<short>"}],"missed":[{"label":"<id>","note":"<where>"}]}]}
Use the Write tool. Include one corrections entry per frame in your batch.

Then return the summary counts (frames reviewed, total boxes kept/relabeled/deleted, missed flagged) and out_file={out_dir}/batch_{first}.json.
```
