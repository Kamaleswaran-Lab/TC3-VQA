# Full review of assembled items

Step: Verification: every item with its four questions, judged against the frame; flagged questions go to the regeneration prompts.

Input: A manifest slice of `{item_id, image_path, concept_readable, anatomy, questions}`.

Output: JSON `{items: [{item_id, mcq_ok, true_concept, anatomy_ok, true_anatomy, q_issues, image_quality, overall, note}]}` written to the batch output file.

The prompt follows as it was run; the batch instructions inside it are handled by scripts/run_prompt_batch.py.

```
You are a strict TCCC QA reviewer doing a FULL census of an image-grounded VQA dataset. For each item, open the frame and critically check the 4 questions against what is actually visible. Be honest and specific.

STEP 1 — load (indices {first}..{last}):
  python3 -c "import json;d=json.load(open('{batch_file}'));print(json.dumps(d[{first}:{last}]))"
Each = {item_id, image_path, concept_readable, anatomy, questions:[{type,question,answer}]} where types are recognition_mcq / doctrine_scene / reasoning / how.

STEP 2 — Read the frame at image_path; look carefully.

STEP 3 — assess (judge from the pixels):
  - mcq_ok: does the recognition_mcq answer match the intervention actually shown? (no→give true_concept)
  - anatomy_ok: is "anatomy" the right body region the intervention is on? (no→true_anatomy)
  - For EACH of doctrine_scene/reasoning/how, list any issue: "telegraph" (question reveals/contains its answer), "so_what" (answer vague/non-actionable e.g. 'reassess the wound'), "off_topic" (answer not relevant to this scene/concept), "noise" (answer has page-header/date junk), "wrong" (doctrine incorrect for the scene). Omit a question if it's fine.
  - image_quality: good | poor (blurry/dark but usable) | unusable
  - overall: "keep" (all good) | "fix" (specific issues, salvageable) | "drop" (image unusable or no real intervention) 

STEP 4 — Write JSON to {out_dir}/batch_{first}.json:
{"items":[{"item_id":"...","mcq_ok":true,"true_concept":null,"anatomy_ok":true,"true_anatomy":null,"q_issues":[{"qtype":"reasoning","issue":"so_what","note":"..."}],"image_quality":"good","overall":"keep","note":"<=15 words"}]}
Use Write, one per item. Set overall="fix" or "drop" only when there is a real problem.

Return n, written, flagged (items with overall != keep), out_file={out_dir}/batch_{first}.json.
```
