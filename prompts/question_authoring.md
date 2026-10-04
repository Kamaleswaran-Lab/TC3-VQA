# Question authoring

Step: Construction: the four answerable questions per scene, with doctrine and reasoning answers copied verbatim from the retrieved passages; assembled and offset-verified by `tc3_vqa/construction/assemble_questions.py`.

Input: A manifest slice of scenes with `item_id`, `concept`, `visible_evidence`, `visual_observation`, `image_path`, `detected`, `anatomy`, `retrieved` passages, `assigned_facet`, `preferred_rank`.

Output: JSON `{items: [{item_id, mcq, doctrine, reasoning}]}` written to the batch output file.

The prompt follows as it was run; the batch instructions inside it are handled by scripts/run_prompt_batch.py.

```
You author DIVERSE, scene-grounded TCCC visual-QA items. Doctrine/reasoning answers MUST be copied VERBATIM from the provided corpus passages (correct-by-construction; never paraphrase).

STEP 1 — load your scenes (indices {first}..{last}):
  python3 -c "import json;d=json.load(open('{batch_file}'))['scenes'];print(json.dumps(d[{first}:{last}]))"
Each scene = {item_id, concept, concept_readable, visible_evidence, visual_observation, image_path, detected, anatomy, retrieved:[{cid,source_id,anchor,text}], assigned_facet, preferred_rank}.

STEP 2 — Read the frame at image_path.

STEP 3 — produce THREE questions:
  - mcq: {"question":"Which TCCC intervention is being performed in this frame?","options":[4 readable interventions incl correct],"answer":"<correct=concept_readable>"}
  - doctrine: **DIVERSITY IS REQUIRED.** Frame the doctrine question around THIS scene's "assigned_facet" ({assigned_facet} style: indication=when/why-indicated, technique=how-performed, placement_or_site=where, sequence_or_next_step=order/after, effectiveness_or_verification=confirm-it-worked, caution_or_contraindication=what-to-avoid). The "answer" = a CONTIGUOUS substring copied EXACTLY from a passage covering that facet. PREFER retrieved[preferred_rank] as the source IF it genuinely contains relevant doctrine for this concept+facet; else use the anchor or the most relevant passage. Quote the SPECIFIC sentence/aspect for the assigned_facet (the anchor often has several sentences — pick the one matching the facet) so that different scenes of the same concept get DIFFERENT doctrine spans. record "cid","source_id","facet".
  - reasoning: a facet DIFFERENT from the doctrine facet (why / next_step / contraindication / consequence); verbatim substring, PREFER a DIFFERENT source. record "cid","source_id","reason_type".
Rules: answers must be exact substrings (str.find) of the chosen passage — copy, don't retype. If a passage isn't truly relevant, do NOT use it (quality over the preference). Questions scene-specific.

STEP 4 — Write JSON to {out_dir}/batch_{first}.json:
{"items":[{"item_id":"...","mcq":{...},"doctrine":{"question":"...","answer":"<verbatim>","cid":"...","source_id":"...","facet":"..."},"reasoning":{"question":"...","answer":"<verbatim>","cid":"...","source_id":"...","reason_type":"..."}}]}
Use Write. VERIFY each verbatim answer is an exact substring before writing.

Return n, written, out_file={out_dir}/batch_{first}.json.
```
