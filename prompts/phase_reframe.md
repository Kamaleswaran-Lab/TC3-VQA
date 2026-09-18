# Reframing doctrine tied to the wrong phase of care

Step: Refinement: tourniquet items whose doctrine or reasoning answer asserted a care-under-fire context the scene does not show. The second prompt is the independent faithfulness check of the proposed rewrite.

Input: One item JSON: `scene_desc` and the subquestions with `question`, `answer`, `source_quote`.

Output: JSON `{item_id, subs: [{qid, new_question, new_answer}]}`; the check returns `{item_id, checks: [{qid, faithful, cuf_present}]}`. Rewrites that fail the check were dropped.

The prompt follows as it was run; the batch instructions inside it are handled by scripts/run_prompt_batch.py.

### Rewrite

```
Reframe TCCC tourniquet doctrine QA so it fits the actual scene. Read the JSON at {batch_file}: it has scene_desc, and subquestions (doctrine_scene / reasoning) each with question, answer, and source_quote (the verbatim corpus span the answer must stay faithful to).

For EACH subquestion, produce new_question and new_answer that:
- REMOVE any "Care Under Fire" / "CUF" / "enemy fire" / "under fire" / phase-specific combat framing, UNLESS scene_desc clearly shows ACTIVE COMBAT (e.g., live fire, smoke + threat). The scenes are overwhelmingly calm training/field demos — frame the doctrine as GENERAL extremity-hemorrhage tourniquet doctrine (true regardless of phase).
- Stay FULLY SUPPORTED by that subquestion's source_quote: keep only the general doctrine the span contains (e.g., extremity hemorrhage is a leading cause of preventable death; apply a tourniquet high/tight above the bleeding site; tighten until bleeding stops). Do NOT add facts that are not in the source_quote. You may DROP the CUF-specific clause and keep the general doctrine.
- Stay clinically correct, concise, and answerable from the depicted tourniquet scene.

Return {item_id, subs:[{qid,new_question,new_answer}]} copying item_id and each qid from the file.
```

### Independent check

```
You are an INDEPENDENT faithfulness checker (you did not write these). Read {batch_file} to get each subquestion's source_quote (the verbatim corpus span).

Proposed new answers: {proposed_answers}

For each qid, judge:
- faithful: is EVERY claim in new_answer supported/entailed by that subquestion's source_quote? (false if it adds any unsupported clinical claim)
- cuf_present: does new_answer still assert a Care-Under-Fire / enemy-fire / active-combat context?

Return {item_id, checks:[{qid,faithful,cuf_present}]}.
```
