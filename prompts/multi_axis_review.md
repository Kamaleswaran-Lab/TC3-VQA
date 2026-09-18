# Multi-axis review of assembled items

Step: Verification: every candidate item after question assembly; refusal items use the second prompt.

Input: A JSON array of items with `item_id`, `concept`, `frame_paths`, `questions`, `candidate_doctrine_answer`, `visual_observation`, `visible_evidence`.

Output: JSON `{verdicts: [...]}` with per-item `frame_shows_concept`, `recog_correct`, `distractors_ok`, `doctrine_accurate`, `doctrine_grounded`, `reasoning_ok`, `image_conditioned`, `verdict` (keep / fix / drop), `issue`; refusal items return `refusal_appropriate` and `issue`.

The prompt follows as it was run; the batch instructions inside it are handled by scripts/run_prompt_batch.py.

### Answerable items

```
You are auditing items from a TCCC (Tactical Combat Casualty Care) visual-QA benchmark. Read the JSON array at {batch_file} (use the Read tool). Each item has: item_id, concept (assigned TCCC concept), frame_paths (image files), questions (a recognition_mcq with options+answer, a doctrine_scene Q+A that should reference the visible scene, and a reasoning Q+A), candidate_doctrine_answer (corpus-templated doctrine), visual_observation, visible_evidence.

For EACH item: FIRST use the Read tool to VIEW the frame image(s) in frame_paths (view all of them, up to 3). Then judge AGAINST WHAT YOU SEE and TCCC doctrine:
- frame_shows_concept: does the frame actually depict the assigned concept? yes / partial / no.
- recog_correct: is the recognition_mcq "answer" the correct intervention for what is actually in the frame?
- distractors_ok: are the other MCQ options visually confusable but clearly NOT also-correct (not gameable, not ambiguous)?
- doctrine_accurate: is the doctrine_scene answer correct per TCCC doctrine? yes / partial / no.
- doctrine_grounded: does the doctrine_scene question reference a detail ACTUALLY VISIBLE in the frame (image-conditioned), not a generic prompt?
- reasoning_ok: is the reasoning question sensible and its answer doctrinally correct?
- image_conditioned: overall, does correctly answering this item REQUIRE looking at the image (vs being answerable blind from text alone)?
- verdict: keep (good) / fix (salvageable wording/grounding issue) / drop (frame does not show the concept, or content is wrong).
- issue: one short clause naming the main problem, or "" if none.

Be strict and honest. Return ONLY the structured output object {verdicts:[...]}, one entry per item, in item_id order.
```

### Refusal items

```
You are auditing REFUSAL-gold items from a TCCC visual-QA benchmark. Read the JSON array at {batch_file} (Read tool). Each item asks about a TCCC procedure but the gold answer is REFUSE: the frame should NOT support answering (it is a static title card, animation/diagram, non-medical scene, or has no casualty/procedure visible).

For EACH item: use the Read tool to VIEW the frame image(s) in frame_paths, then judge:
- refusal_appropriate: TRUE if REFUSE is correct (the frame genuinely does NOT depict the queried procedure / is non-informative). FALSE if the frame actually DOES clearly show the queried procedure (so refusing would be wrong).
- issue: one short clause, or "".

Return ONLY the structured output object {verdicts:[{item_id, refusal_appropriate, issue}]}, one entry per item.
```
