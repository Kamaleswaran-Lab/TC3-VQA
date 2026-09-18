# Removing uncertain body-part references from question stems

Step: Refinement: question stems that named a specific extremity or torso region the frame could not confirm.

Input: A manifest slice of `{item_id, concept_readable, questions_to_fix: [{qtype, question}]}`.

Output: JSON `{items: [{item_id, fixes: [{qtype, new_question}]}]}` written to the batch output file.

The prompt follows as it was run; the batch instructions inside it are handled by scripts/run_prompt_batch.py.

```
Lightly rewrite TCCC question STEMS to remove UNCERTAIN specific body-part claims. The auto-labeled body part is often wrong (a "leg wound" may be an arm/torso). Per policy: unless the exact body part is essential AND certain, use a generic reference so the question is never wrong about anatomy.

STEP 1 — load (indices {first}..{last}):
  python3 -c "import json;d=json.load(open('{batch_file}'))[{first}:{last}];print(json.dumps(d))"
Each = {item_id, concept_readable, questions_to_fix:[{qtype, question}]}.

STEP 2 — for EACH question, produce a "new_question" that:
  - Replaces specific body-part words (leg, arm, thigh, forearm, upper/lower arm/leg, shin, calf, torso, abdomen, wrist, knee, ankle, elbow, shoulder) with a GENERIC reference: "the casualty", "the casualty's body", "the wound", "the bleeding wound", "the affected limb", or "the injury site" — whichever reads naturally and keeps the question's meaning.
  - Keeps the rest of the question (the clinical intent, the concept) intact. Do NOT change what is being asked. Do NOT touch any answer.
  - Stays natural English, not awkward (e.g. "packed into the bleeding leg wound" -> "packed into the bleeding wound"; "on this casualty's leg" -> "on this casualty"; "this torso wound" -> "this wound").
  - If a body part is intrinsic to the intervention and reliable (e.g. "nostril" for nasopharyngeal airway, "neck" for cricothyroidotomy, "chest" for chest seal/needle decompression), you MAY keep it. Only genericize the uncertain extremity/torso references.

STEP 3 — Write JSON to {out_dir}/batch_{first}.json:
{"items":[{"item_id":"...","fixes":[{"qtype":"how","new_question":"..."}, ...]}]}
Use Write, one entry per item, one fix per question_to_fix.

Return n, written (questions rewritten), out_file={out_dir}/batch_{first}.json.
```
