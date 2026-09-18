# Prompts

These are the prompts of the construction steps that were run by a vision-language model with file access.
The model read the frames and the batch file itself and wrote its JSON back to the batch output file. The prompt
text is reproduced as it was run; template variables appear as placeholders such as {batch_file}. Each file starts
with a note on the step, its input and its output.

scripts/run_prompt_batch.py replays a prompt through the Messages API. It attaches the frames as images, appends the
batch items as JSON and stores the reply as the batch output, so the file-handling instructions inside a prompt are
carried out by the runner.

In the order they were applied:

    visual_audit.md            independent visual audit of the candidate pool
    box_review.md              review of equipment boxes
    anatomy.md                 observed body region
    concept_audit.md           concept audit after re-perception with verified boxes
    question_authoring.md      recognition, doctrine and reasoning questions
    how_questions.md           procedural questions
    refusal_questions.md       questions the frames cannot answer
    multi_axis_review.md       review of assembled items
    safe_repair.md             repair of review-flagged items
    frame_quality_audit.md     frame quality and caption leakage
    caption_boxes.md           text boxes for caption masking
    phase_reframe.md           doctrine tied to the wrong phase of care
    full_census.md             full review of every item
    regenerate_flagged.md      regeneration of flagged questions
    regenerate_restating.md    regeneration of questions that restate their answer
    regenerate_flagged_final.md  last regeneration pass
    anatomy_concept_audit.md   concept and body-region disagreements
    generic_body_parts.md      removal of uncertain body parts from question stems

Two prompts were not kept: the one that mined doctrine facets and the one that wrote refusal-item metadata.
construction/build_facet_inventory.py reads the facet list from a file instead.
