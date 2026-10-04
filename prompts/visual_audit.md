# Independent visual audit of the candidate pool

Step: Verification, step (4): one call per candidate item, blind to the perception model's reasoning.

Input: Per item: frame paths, the assigned concept, and the intended question.

Output: JSON with `concept_visible` (yes / partial / no), `answerable` (bool), `what_you_see`, `confidence` (high / medium / low). Merged by `tc3_vqa/perception/merge_audit.py`.

The prompt follows as it was run; the batch instructions inside it are handled by scripts/run_prompt_batch.py.

```
You are an independent visual adjudicator for a Tactical Combat Casualty Care (TCCC) VQA benchmark.
Open and look at these frame image file(s) with the Read tool (they are from ONE moment of a video, in time order):
  frame 0: {frame_path_0}
  frame 1: {frame_path_1}
  ...

Candidate TCCC concept assigned to this moment: "{concept}".
Intended question: "{question}"

Judge FRESH and STRICTLY from the pixels only. Do NOT assume the assignment is correct.
- concept_visible: is THAT specific procedure clearly and unambiguously being performed/visible? 'no' if the frame is a wide/occluded/ambiguous scene, a different procedure, a talking head, a slide, or if you cannot actually see the named intervention. 'partial' if suggested but not clearly shown.
- For drug concepts (TXA, IV/IO fluids) remember you usually CANNOT tell which drug/fluid from pixels: if all you see is a generic syringe/line, concept_visible is at most 'partial'.
- answerable: could the intended question be fairly grounded in what is actually visible?
Return the structured verdict.
```
