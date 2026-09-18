# Regeneration of restating or noisy questions

Step: Refinement: doctrine and reasoning questions that telegraphed their answer or whose quoted span contained page-header noise.

Input: A manifest slice with `regen_types`, `retrieved` passages, and `keep_questions`.

Output: JSON `{items: [{item_id, doctrine_scene, reasoning}]}` written to the batch output file.

The prompt follows as it was run; the batch instructions inside it are handled by scripts/run_prompt_batch.py.

```
You FIX flagged TCCC visual-QA questions that are low quality. Each flagged question is either (1) RESTATING — the question telegraphs/contains its own answer (e.g. "what does doctrine caution about packing chest/abdomen wounds?"→"Avoid packing wounds in the chest or abdomen"), or (2) NOISY — the verbatim answer has a page-header bled into it ("TCCC Guidelines 2024", a date, "RED text indicates"). Regenerate them better. Answers MUST stay VERBATIM from the provided passages (correct-by-construction).

STEP 1 — load your items (indices {first}..{last}):
  python3 -c "import json;d=json.load(open('{batch_file}'));print(json.dumps(d[{first}:{last}]))"
Each = {item_id, concept, concept_readable, visible_evidence, image_path, regen_types:[...], retrieved:[{cid,source_id,anchor,text}], keep_questions:[the item's OTHER questions you must NOT duplicate]}.

STEP 2 — Read the frame at image_path.

STEP 3 — for EACH type in regen_types ("doctrine_scene" and/or "reasoning"), produce a replacement:
  - The QUESTION must NOT contain or telegraph its answer; it should require real doctrinal knowledge to answer (a learner could not answer it just by rereading the question).
  - The ANSWER = a CONTIGUOUS substring copied EXACTLY from ONE passage's text. Choose a CLEAN span with NO page header / date / "Guidelines 2024" / "RED text" noise mid-quote (if the best sentence is split by such noise, pick a different clean sentence). Must be str.find-able in that passage.
  - Make it SUBSTANTIVE: avoid the over-used vague "reassess the wound" / "convert to dressing" answers — pick a more informative doctrine span and a different facet from keep_questions (no duplication of the kept questions).
  - record cid, source_id, and facet (doctrine) or reason_type (reasoning).

STEP 4 — Write JSON to {out_dir}/batch_{first}.json:
{"items":[{"item_id":"...","doctrine_scene":{"question":"...","answer":"<verbatim>","cid":"...","source_id":"...","facet":"..."},"reasoning":{"question":"...","answer":"<verbatim>","cid":"...","source_id":"...","reason_type":"..."}}]}
Only include the keys present in that item's regen_types. Use Write. VERIFY each answer is an exact substring before writing.

Return n, written (questions regenerated), out_file={out_dir}/batch_{first}.json.
```
