# Observed body region

Step: Annotation: one region per answerable item, judged from the frame.

Input: A manifest slice of `{item_id, image_path, concept, concept_readable, detected, visible_evidence}`.

Output: JSON `{items: [{item_id, anatomy, confidence, reason}]}` written to the batch output file.

The prompt follows as it was run; the batch instructions inside it are handled by scripts/run_prompt_batch.py.

```
You determine the OBSERVED anatomical body region where a TCCC intervention is performed on the CASUALTY. Judge by the actual BODY PART (human anatomy), NOT by where it sits in the frame (a casualty lying down still has an ARM = upper_extremity even if the arm is low in the image). Ignore the provider's own body; judge the casualty/manikin being treated.

STEP 1 — load your items (indices {first}..{last}):
  python3 -c "import json;d=json.load(open('{batch_file}'));print(json.dumps(d[{first}:{last}]))"
Each = {item_id, image_path, concept, concept_readable, detected, visible_evidence}.

STEP 2 — Read the frame at image_path; look at where on the casualty's body the intervention/equipment/hands are.

STEP 3 — choose ONE region:
  head_face, neck, chest, abdomen, pelvis_groin, upper_extremity (arm: shoulder/upper-arm/forearm/hand), lower_extremity (leg: thigh/knee/shin/foot), back, multiple_generic (a wrap/blanket or treatment spanning several regions), unclear (close-up/ambiguous so the body region truly cannot be told).
Give a <=12-word reason citing the visible body landmark.

STEP 4 — Write JSON to {out_dir}/batch_{first}.json:
{"items":[{"item_id":"...","anatomy":"<region>","confidence":"high|medium|low","reason":"..."}]}
Use Write, one per item.

Return n, written, out_file={out_dir}/batch_{first}.json.
```
