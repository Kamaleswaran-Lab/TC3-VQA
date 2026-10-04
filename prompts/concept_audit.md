# Concept audit after re-perception

Step: Concept assignment: items whose re-perceived concept differed from the original; applied by `tc3_vqa/perception/apply_concept_audit.py`. Verdict values are `original`, `reperceived`, `other`, or `none`.

Input: A manifest slice of `{item_id, file, previous_concept, concept, detected, observation}` and the closed concept list.

Output: JSON `{audits: [{item_id, correct, verdict, reason}]}` written to the batch output file.

The prompt follows as it was run; the batch instructions inside it are handled by scripts/run_prompt_batch.py.

```
You are a TCCC expert auditing which intervention a video frame ACTUALLY shows. Judge ONLY from what is visible in the pixels.

STEP 1 — get your batch (items {first}..{last}):
  python3 -c "import json;d=json.load(open('{batch_file}'));print(json.dumps(d[{first}:{last}]))"
Each entry = {item_id, file, previous_concept, concept, detected, observation}.

STEP 2 — for each entry, Read the image at {frames_dir}/<file> and look carefully.

STEP 3 — decide which ONE concept the frame actually depicts, choosing from this closed list:
{concepts}
For each item set "correct" to the concept_id you judge is truly shown, or "none" if no concept is clearly depicted (ambiguous/black/non-procedural). Then set "verdict":
  - "original" if correct == previous_concept
  - "reperceived" if correct == concept
  - "other" if correct is a different concept id
  - "none" if nothing clear
Give a <=12-word reason naming the visual evidence.

STEP 4 — Write JSON to {out_dir}/batch_{first}.json with shape:
{"audits":[{"item_id":"...","correct":"<concept_id|none>","verdict":"original|reperceived|other|none","reason":"..."}]}
Use the Write tool, one entry per item.

Then return counts: n, previous_correct (verdict original), reperceived_correct (verdict reperceived), other, none_correct (verdict none), and out_file={out_dir}/batch_{first}.json.
```

The `{concepts}` list as used:

tourniquet_application: windlass limb tourniquet (CAT) being placed/tightened on an extremity
tourniquet_conversion: a tourniquet ALREADY applied, being re-examined/loosened/replaced with a dressing (not first placement)
wound_packing: hemostatic gauze packed into an open wound / two-handed direct pressure into a bleeding wound
junctional_hemorrhage: wound at groin/axilla/pelvic junction; junctional tourniquet device, or gauze+pressure at high-thigh/groin
chest_seal: adhesive/occlusive seal applied to an open chest/torso wound
needle_decompression: large-bore needle/catheter into the chest wall
nasopharyngeal_airway: soft nasal trumpet tube at/in the nostril, or airway positioning of an unconscious casualty
surgical_airway_cric: incision / tube at the anterior neck (cricothyroidotomy)
hypothermia_prevention: thermal blanket/wrap (HPMK/Blizzard) around the casualty
txa_administration: drawing/giving a syringe of medication via IV/IO
iv_io_access: IV catheter into arm / IO needle into bone / IV bag+line being set up
tccc_documentation: writing on a TCCC/DD-1380 casualty card with a marker
