# Procedural HOW questions

Step: Construction: one HOW question per scene, answer copied verbatim from a procedural passage; attached by `tccc_vqa/construction/add_how_questions.py`.

Input: A manifest slice of scenes with the retrieved passages (as for question authoring).

Output: JSON `{items: [{item_id, how: {question, answer, cid, source_id}}]}` written to the batch output file.

The prompt follows as it was run; the batch instructions inside it are handled by scripts/run_prompt_batch.py.

```
You add ONE actionable "HOW" question per scene for a TCCC visual-QA dataset that trains a VLM to help an INEXPERIENCED medic or a ROBOT actually PERFORM the right intervention. The answer MUST be copied VERBATIM from the provided corpus passages (correct-by-construction).

WHY HOW: the dataset already has what/where/why questions; it lacks HOW. A good HOW question is recognition-grounded ("given what is seen, HOW should the responder perform/secure/place/advance ... ?") and its answer is the actual PROCEDURAL doctrine (the steps/technique), not a vague "reassess" restatement.

STEP 1 — load your scenes (indices {first}..{last}):
  python3 -c "import json;d=json.load(open('{batch_file}'))['scenes'];print(json.dumps(d[{first}:{last}]))"
Each scene = {item_id, concept, concept_readable, visible_evidence, visual_observation, image_path, detected, anatomy, retrieved:[{cid,source_id,anchor,text}]}.

STEP 2 — Read the frame at image_path.

STEP 3 — produce ONE "how" question:
  - "how": {"question":"<scene-grounded: how should the responder perform/complete/secure/place this intervention here?>","answer":"<CONTIGUOUS substring copied EXACTLY from ONE passage describing the PROCEDURE/technique STEPS (prefer the anchor or a technique passage)>","cid":"...","source_id":"..."}
Rules: the answer = actionable procedural doctrine (steps/how-to), VERBATIM (exact substring, str.find-able; copy don't retype). The question must ask HOW (technique/sequence of doing it), grounded in this scene. If no passage has procedural steps, quote the most actionable technique doctrine available. Avoid trivially restating the question.

STEP 4 — Write JSON to {out_dir}/batch_{first}.json:
{"items":[{"item_id":"...","how":{"question":"...","answer":"<verbatim>","cid":"...","source_id":"..."}}]}
Use Write. VERIFY each answer is an exact substring of its passage before writing.

Return n, written, out_file={out_dir}/batch_{first}.json.
```
