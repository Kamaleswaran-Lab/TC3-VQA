# Adjudication of concept-anatomy disagreements

Step: Verification: items whose labelled intervention and body region were inconsistent; applied by `tccc_vqa/construction/apply_anatomy_audit.py`.

Input: A manifest slice of `{item_id, image_path, current_concept, current_anatomy, visible_evidence, detected}`.

Output: JSON `{items: [{item_id, true_concept, true_anatomy, verdict, reason}]}` written to the batch output file.

The prompt follows as it was run; the batch instructions inside it are handled by scripts/run_prompt_batch.py.

```
You adjudicate frames where the labeled CONCEPT (intervention) and the labeled ANATOMY (body region) disagree. Judge ONLY from the pixels which is wrong.

STEP 1 — load (indices {first}..{last}):
  python3 -c "import json;d=json.load(open('{batch_file}'));print(json.dumps(d[{first}:{last}]))"
Each = {item_id, image_path, current_concept, current_anatomy, visible_evidence, detected}.

STEP 2 — Read the frame at image_path and look carefully.

STEP 3 — decide the TRUE intervention concept (from: {concepts}; or "none" if no clear intervention) and the TRUE body region (head_face/neck/chest/abdomen/pelvis_groin/upper_extremity/lower_extremity/back/multiple_generic/unclear). Then set:
  - "true_concept", "true_anatomy"
  - "verdict": "ok" (both current labels right) | "fix_anatomy" (concept right, anatomy wrong) | "fix_concept" (anatomy right, concept wrong) | "fix_both" | "refuse" (no clear intervention) | "ambiguous"
  - "reason" (<=15 words citing the visible evidence)

STEP 4 — Write JSON to {out_dir}/batch_{first}.json:
{"items":[{"item_id":"...","true_concept":"...","true_anatomy":"...","verdict":"...","reason":"..."}]}
Use Write, one per item. Return n, written, out_file={out_dir}/batch_{first}.json.
```

The `{concepts}` list as used:

tourniquet_application, tourniquet_conversion, wound_packing, junctional_hemorrhage (groin/axilla/pelvis device or gauze), chest_seal, needle_decompression (needle into chest), nasopharyngeal_airway, surgical_airway_cric (incision/tube at anterior neck), hypothermia_prevention, iv_io_access (IV/IO incl sternal/humeral IO), txa_administration, tccc_documentation
