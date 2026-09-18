# Final regeneration pass

Step: Refinement: questions flagged a second time after regeneration.

Input: As for `regenerate_flagged.md`.

Output: As for `regenerate_flagged.md`.

The prompt follows as it was run; the batch instructions inside it are handled by scripts/run_prompt_batch.py.

```
Final-pass fix of low-quality TCCC VQA questions (a reviewer flagged them twice). Read the frame, pick the doctrine that GENUINELY and SPECIFICALLY fits THIS scene, and write a clean, substantive, non-telegraphing question. Answers VERBATIM from passages.

STEP 1 — load (indices {first}..{last}):
  python3 -c "import json;d=json.load(open('{batch_file}'))[{first}:{last}];print(json.dumps(d))"
Each = {item_id, concept, concept_readable, visible_evidence, image_path, regen_types, issues (prior flags incl 'wrong'=doctrine not right for scene), retrieved:[{cid,source_id,anchor,text}], keep_answers}.

STEP 2 — Read the frame; note the exact situation.

STEP 3 — for each type in regen_types: read ALL retrieved passages, choose the span whose doctrine is CORRECT and SPECIFIC for this exact scene (fix prior 'wrong'/'off_topic'). ANSWER = exact contiguous substring (clean, no header/date noise, substantive — no vague 'reassess'). QUESTION = scene-grounded, must not telegraph the answer, distinct from keep_answers. record cid, source_id, facet/reason_type.

STEP 4 — Write JSON to {out_dir}/batch_{first}.json:
{"items":[{"item_id":"...","doctrine_scene":{...},"reasoning":{...},"how":{...}}]} (only regen_types keys). Use Write. VERIFY exact substring.

Return n, written, out_file={out_dir}/batch_{first}.json.
```
