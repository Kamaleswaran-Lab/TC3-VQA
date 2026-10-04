# Regeneration of flagged questions

Step: Refinement: questions the full review flagged (off-topic, vague, noisy, wrong, or telegraphing); applied by `tc3_vqa/construction/apply_regeneration.py`.

Input: A manifest slice with `regen_types`, `issues`, `retrieved` passages, and the item's other questions (`keep_answers`).

Output: JSON `{items: [{item_id, doctrine_scene, reasoning, how}]}` with only the regenerated types, written to the batch output file.

The prompt follows as it was run; the batch instructions inside it are handled by scripts/run_prompt_batch.py.

```
You FIX TCCC visual-QA questions a reviewer flagged. The biggest flaw is OFF_TOPIC: the verbatim doctrine answer is real but NOT specific to THIS scene (generic concept doctrine pasted on). Also fix so_what (vague/non-actionable), noise (page-header in quote), wrong (doctrine not right for the scene), telegraph (question reveals its answer). Answers MUST stay VERBATIM from the provided passages (correct-by-construction).

STEP 1 — load (indices {first}..{last}):
  python3 -c "import json;d=json.load(open('{batch_file}'))[{first}:{last}];print(json.dumps(d))"
Each = {item_id, concept, concept_readable, visible_evidence, image_path, regen_types:[...], issues:[{qtype,issue,note}], retrieved:[{cid,source_id,anchor,text}], keep_answers:[other kept Qs — don't duplicate]}.

STEP 2 — Read the frame at image_path. Note the SPECIFIC situation (body region, severity, what stage, what is visible).

STEP 3 — for EACH type in regen_types ("doctrine_scene"/"reasoning"/"how"):
  - Pick the passage span whose doctrine GENUINELY APPLIES to THIS specific scene (not generic). If several passages fit, choose the one most specific to what's visible. Read all retrieved passages and judge relevance — do NOT just take the anchor if a more scene-fitting span exists.
  - ANSWER = CONTIGUOUS substring copied EXACTLY from that passage, a CLEAN span (no "TCCC Guidelines 2024"/date/"RED text" header mid-quote), SUBSTANTIVE (not vague 'reassess'/'convert' restatement; for 'how' = concrete procedural steps), str.find-able.
  - QUESTION = scene-grounded, must NOT telegraph the answer (a learner can't answer it by rereading the question), different from keep_answers.
  - record cid, source_id, facet (doctrine) / reason_type (reasoning).
  If truly no retrieved passage is scene-relevant for a type, still pick the closest on-concept clean span (do not invent).

STEP 4 — Write JSON to {out_dir}/batch_{first}.json:
{"items":[{"item_id":"...","doctrine_scene":{"question":"...","answer":"<verbatim>","cid":"...","source_id":"...","facet":"..."},"reasoning":{...},"how":{"question":"...","answer":"<verbatim>","cid":"...","source_id":"..."}}]}
Only include keys in that item's regen_types. Use Write. VERIFY each answer is an exact substring before writing.

Return n, written, out_file={out_dir}/batch_{first}.json.
```
