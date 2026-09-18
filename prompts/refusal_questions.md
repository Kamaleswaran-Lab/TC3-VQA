# In-domain unanswerable questions

Step: Refusal construction: one call per source frame set; the second prompt is the independent check of each candidate question.

Input: One item JSON: `src_item`, `concept`, `frame_paths`.

Output: JSON `{src_item, question, unanswerable_reason, target_category}`; the check returns `{src_item, answerable_from_frame, on_topic, note}`. Questions judged answerable or off-topic were discarded (`tccc_vqa/construction/clean_refusal.py`).

The prompt follows as it was run; the batch instructions inside it are handled by scripts/run_prompt_batch.py.

### Question

```
Create ONE in-domain-UNANSWERABLE TCCC question for this real casualty-care frame.

STEP 1: Read {batch_file} (has src_item, concept, frame_paths). STEP 2: Read EVERY frame image. STEP 3: Look carefully at what IS and ISN'T visible.

Write a question that (a) is a PLAUSIBLE TCCC question a responder might ask about THIS specific scene (reference what is visible so it's tempting), BUT (b) CANNOT be answered from the frame's visual evidence — the needed information is simply not present/determinable.
Good unanswerable targets: vital signs (heart rate, blood pressure, SpO2), elapsed time / how long ago, blood-loss volume, the status of an off-frame body part, a medication's name or dose that isn't legible, the OUTCOME of a step not yet shown, patient history/mechanism. Avoid absurd or generic questions — it must look answerable to a careless model.

Return {src_item (copy), question, unanswerable_reason (what info is missing and why it's not in the frame), target_category}.
```

### Independent check

```
Independent check of a candidate UNANSWERABLE question (you did not write it).
STEP 1: Read {batch_file} for frame_paths. STEP 2: Read EVERY frame. STEP 3: Judge the question below against the visual evidence ONLY.

Question: {question}

- answerable_from_frame: could a careful clinician answer this from the frame's visual content alone? We WANT this false (truly unanswerable). Set true if any reasonable answer is visible/determinable.
- on_topic: is it a plausible, scene-relevant TCCC question (not absurd, not generic)?
Return {src_item, answerable_from_frame, on_topic, note}.
```
