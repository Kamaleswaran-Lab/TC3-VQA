# Pipeline, step by step

Every script that produced the released dataset, in the order it was run. The short version is in the
README; this file is the complete record.

Steps marked "prompt" were run by a vision-language model that read the frames and batch files itself (Claude Opus 4.8). Their prompts are in prompts/ and can be replayed with
`python scripts/run_prompt_batch.py prompts/<step>.md manifest.jsonl --out-dir <dir> --frames-dir <frames>`.
Every prompt step writes batch_<k>.json files that the next script reads.

1. `python -m tc3_vqa.corpus.build_documents` downloads the registered documents and chunks them at headings.
2. `python -m tc3_vqa.corpus.glossary`, `pubmed --target tccc` and `arxiv` add the glossary and the auxiliary literature.
3. `python -m tc3_vqa.corpus.build_index` writes the BGE-large embeddings and the FAISS index.
4. `python -m tc3_vqa.frame_select.select_frames --video <mp4> --video-id <id> --out-dir <frames>` writes candidate frames per video.
5. `python -m tc3_vqa.frame_select.score_frames --frames-dir <frames> --video-list <txt>` keeps windows that show an intervention.
6. `python -m tc3_vqa.perception.perceive_concepts` assigns one concept, a description and a confidence per window.
7. `python -m tc3_vqa.perception.concept_cross_check` re-checks each label with the same model under a strict prompt.
8. prompt `visual_audit.md`, then `perception.merge_audit` and `perception.split_by_audit` set review tiers and set aside unsupported labels.
9. `annotation.detect_vlm`, `detect_gdino`, `detect_consensus`, `detect_verify_vlm` and `detect_sam` propose, cross-check and refine equipment boxes.
10. prompt `box_review.md`, then `annotation.apply_box_review`, `recall_detect` and `export_coco` write the detection layer.
11. `annotation.answer_region` and `answer_region_check` write the answer-region boxes; prompt `anatomy.md` gives the body region.
12. `perception.reperceive_with_boxes`, prompt `concept_audit.md` and `perception.apply_concept_audit` reassess each concept with the verified boxes.
13. `construction.embed_scene_queries` and `construction.select_passages` retrieve the doctrine passages of each scene.
14. `construction.build_facet_inventory <mined_facets.json>` verifies the doctrine facets against the corpus.
15. prompt `question_authoring.md`, then `construction.assemble_questions` matches every answer to its exact corpus span.
16. prompt `how_questions.md`, then `construction.add_how_questions` adds the procedural question.
17. prompt `refusal_questions.md`, then `construction.clean_refusal` builds the refusal items.
18. `verification.nli_gate_concept <answers.jsonl>` keeps answers whose every sentence is entailed by the concept corpus.
19. prompts `multi_axis_review.md`, `safe_repair.md`, `frame_quality_audit.md`, `caption_boxes.md` and `phase_reframe.md`, then `annotation.apply_caption_masks`, review the assembled items and mask captions that reveal answers.
20. prompt `full_census.md`, then the `regenerate_*.md` prompts and `construction.apply_regeneration`, replace flagged questions.
21. prompt `anatomy_concept_audit.md`, then `construction.apply_anatomy_audit`; prompt `generic_body_parts.md` removes uncertain body parts from question stems.
22. `consensus.consensus_perceive --model <hf id> --tag <tag>` runs one recognition voter; `consensus_collect` reads the API-served voter; `consensus_analyze` writes the vote per item.
23. `release.fetch_source_meta` records channel, license and availability of every source video; `release.check_source_availability` re-checks reachability before a release.
24. `adjudication.gold_notes` copies the raters' notes into the anonymized adjudication file.
25. `release.frame_features` computes frozen image-encoder features of every frame on a GPU node.
26. `release.build_release` assembles the full layout and `release.build_deposit` writes the public package.
