# Repair of review-flagged items

Step: Refinement: items the multi-axis review marked as fixable.

Input: A JSON array of flagged items with their questions, the cited doctrine text, the observation, and the audit issue.

Output: JSON `{repairs: [{item_id, action, questions, change_note, residual_issue}]}`; clinical-judgement cases are left unchanged with `action = leave_for_clinician`.

The prompt follows as it was run; the batch instructions inside it are handled by scripts/run_prompt_batch.py.

```
You are repairing flagged items from a TCCC (Tactical Combat Casualty Care) visual-QA benchmark before clinician handoff. Read the JSON array at {batch_file} (Read tool). Each item: item_id, concept, frame_paths, candidate_doctrine_answer (the corpus-cited doctrine text for this concept), visual_observation, audit_issue (the problem found in a prior audit), questions (array of 3: a recognition_mcq {qid,type,question,options,answer}; a doctrine_scene {qid,type,question,answer,source,provenance,faithful}; a reasoning {qid,type,question,answer,reason_type}).

For EACH item: FIRST use the Read tool to VIEW the frame image(s) in frame_paths. Then repair ONLY per these SAFE rules, guided by audit_issue:
1. If the doctrine_scene "answer" merely restates/describes the scene instead of answering its own "question": rewrite "answer" to directly, concisely answer that question (1-2 sentences), grounded in a visible detail, using ONLY facts present in candidate_doctrine_answer (do NOT introduce outside facts). Keep question, source, provenance, faithful, qid, type unchanged.
2. If any "question" (doctrine_scene or reasoning) asserts a body site (e.g., groin/inguinal/junctional), a "care under fire"/"on fire" setting, or any specific detail NOT verifiable in the frame: edit that question to REMOVE the unverified assertion (do not add new claims); keep it answerable and grounded in what IS visible.
3. If a recognition_mcq distractor is also-correct or genuinely ambiguous given the frame: replace ONLY that one option with a clearly-different but visually-confusable TCCC concept that is NOT correct; keep exactly 4 options including the unchanged correct "answer".

Do NOT edit when the issue is a clinical-judgment call: a contraindication question (e.g., packing a chest/torso wound) OR "concept only partially shown / weak frame". For those set action="leave_for_clinician" and return questions UNCHANGED. If nothing needs changing, action="ok_as_is".

Return ONLY structured output {repairs:[{item_id, action, questions:[the full corrected 3-question array, preserving all unchanged sub-fields], change_note, residual_issue}]}, one per item in item_id order. ALWAYS return the full questions array.
```
