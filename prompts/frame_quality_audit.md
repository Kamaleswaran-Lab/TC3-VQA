# Frame quality and caption-leakage audit

Step: Refinement: one call per item; items judged `drop` for caption leakage were recovered by masking the caption.

Input: One item JSON: `item_id`, `concept`, `frame_paths`, `questions` (each with its answer and cited doctrine).

Output: JSON with `image_informative`, `concept_match`, `overlay_leakage`, `recognition_ok`, `doctrine_ok`, `reasoning_ok`, `verdict` (keep / fix / drop), `drop_reasons`, `notes`.

The prompt follows as it was run; the batch instructions inside it are handled by scripts/run_prompt_batch.py.

```
You are auditing ONE Tactical Combat Casualty Care (TCCC) visual-QA item for a clinical benchmark. Quality over quantity — when in doubt, DROP.

STEP 1: Read the JSON file at {batch_file}. It contains: item_id, the assigned `concept`, `frame_paths` (image files), and `questions` (a recognition_mcq with options, a doctrine_scene, and a reasoning question; each has the intended `answer`, and doctrine/reasoning include the `cited_doctrine` verbatim span).
STEP 2: Read EVERY image in frame_paths (use the Read tool on each path). Actually look at the pixels.
STEP 3: Judge strictly and return the schema, with item_id copied from the file.

Criteria:
- image_informative: is the assigned concept's key visual (the device / anatomy / action) clearly identifiable in at least one frame? A clear training MANIKIN counts as informative IF the procedure is unambiguous. NOT informative: title/slide cards, talking-head/instructor-only shots, far or motion-blurred shots where the action isn't legible, transition/black frames, frames showing only unrelated background.
- concept_match: does the frame actually depict the assigned concept? (e.g., concept=tourniquet_application but the strap is actually a litter/drag strap → no). yes / partial / no.
- overlay_leakage: TRUE if any on-screen text overlay reveals the answer to any of the questions (a viewer could answer from the burned-in text without interpreting the image). This is a DROP condition.
- recognition_ok / doctrine_ok / reasoning_ok: for each question, is it answerable from the image AND is the given answer correct and applicable to THIS specific scene?
- verdict: 'keep' if all good; 'fix' if informative AND concept matches AND no leakage but a QA axis is imperfect/awkward (salvageable by editing); 'drop' if NOT image_informative, OR concept_match=no, OR overlay_leakage, OR the item is fundamentally broken/meaningless.

List concrete drop_reasons. Be decisive; lean toward drop for noise.
```
