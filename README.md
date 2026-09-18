# TCCC-VQA

Code that built the TCCC-VQA dataset and ran the evaluations in the accompanying paper. TCCC-VQA is a visual
question answering dataset for Tactical Combat Casualty Care. Each item pairs frames from a public TCCC video with a
recognition question, a doctrine question, a clinical-reasoning question and a procedural question, or with a
question the frames cannot answer. Doctrine, reasoning and procedural answers are verbatim passages of public TCCC
documents, cited by character offset.

The dataset is published separately at https://doi.org/10.5281/zenodo.22818562. This repository holds the scripts and prompts
that produced it and the scripts that scored the baseline models.

## Layout

    tccc_vqa/paths.py       working directories, read from environment variables
    tccc_vqa/frame_select   shot detection, frame sampling and filtering, window scoring
    tccc_vqa/corpus         doctrine corpus: sources, download, chunking, embeddings, index
    tccc_vqa/perception     concept assignment and its checks
    tccc_vqa/annotation     equipment boxes, answer regions, caption masks
    tccc_vqa/construction   passage selection, question assembly, regeneration
    tccc_vqa/verification   claim-level entailment gate
    tccc_vqa/consensus      recognition consensus of six models
    tccc_vqa/release        release assembly and the public package
    tccc_vqa/eval           baseline evaluation, controls, release-wide audit
    tccc_vqa/ablation       reference answers versus model-authored answers
    tccc_vqa/analysis       figures and table checks
    tccc_vqa/adjudication   the physician adjudication sample
    prompts/                prompts of the steps run through a vision-language model with file access
    scripts/                run_prompt_batch.py, which replays those prompts through the Messages API
    configs/                SLURM examples for the GPU steps

## Installation

Python 3.10 or later. The GPU steps were run with vLLM 0.15 on NVIDIA H200 GPUs. The entailment gate uses
MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli and retrieval uses BAAI/bge-large-en-v1.5.

    pip install -r requirements.txt
    pip install -e .

Set the working directories before running anything (see configs/README.md). Scripts run as modules, for example
`python -m tccc_vqa.frame_select.select_frames --help`.

## Pipeline

Steps marked "prompt" were run by a vision-language model that read the frames and batch files itself; Claude Opus 4.8
was used. Their prompts are in prompts/ and can be replayed with
`python scripts/run_prompt_batch.py prompts/<step>.md manifest.jsonl --out-dir <dir> --frames-dir <frames>`.
Every prompt step writes batch_<k>.json files that the next script reads.

1. `python -m tccc_vqa.corpus.build_documents` downloads the registered documents and chunks them at headings.
2. `python -m tccc_vqa.corpus.glossary`, `pubmed --target tccc` and `arxiv` add the glossary and the auxiliary literature.
3. `python -m tccc_vqa.corpus.build_index` writes the BGE-large embeddings and the FAISS index.
4. `python -m tccc_vqa.frame_select.select_frames --video <mp4> --video-id <id> --out-dir <frames>` writes candidate frames per video.
5. `python -m tccc_vqa.frame_select.score_frames --frames-dir <frames> --video-list <txt>` keeps windows that show an intervention.
6. `python -m tccc_vqa.perception.perceive_concepts` assigns one concept, a description and a confidence per window.
7. `python -m tccc_vqa.perception.concept_cross_check` re-checks each label with the same model under a strict prompt.
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
23. `release.fetch_source_meta` records channel, licence and availability of every source video.
24. `release.build_release` assembles the full layout and `release.build_deposit` writes the public package.

## Evaluation

    python -m tccc_vqa.eval.build_eval_inputs
    sbatch --export=ALL,MODEL=<hf id>,TAG=<tag> configs/baseline_eval.sbatch
    TCCC_VQA_RUNS=<runs> python -m tccc_vqa.eval.score_eval <tag>
    python -m tccc_vqa.eval.score_doctrine <runs>/doctrine_<tag>.jsonl
    python -m tccc_vqa.eval.build_doctrine_mcq --qtype easy
    python -m tccc_vqa.eval.build_control_inputs && python -m tccc_vqa.eval.score_controls
    python -m tccc_vqa.eval.release_audit --model <hf id> --tag <tag> && python -m tccc_vqa.eval.release_audit_analyze
    python -m tccc_vqa.eval.build_yolo_dataset <out dir> && yolo detect train model=yolov8s.pt data=<out dir>/data.yaml epochs=100 imgsz=640

The comparison with model-authored QA is in tccc_vqa/ablation: `gen_conventional` (open models) and `prep_answer_only`
with `collect_batches` (API-served model) produce the arms, `build_judge_inputs` builds blinded judge inputs,
`judge` runs the local judges, and `analyze`, `figure_data` and `paired_rubric` aggregate them.

`python -m tccc_vqa.analysis.verify_tables` recomputes every table cell of the paper; `make_plots`,
`ablation_examples` and `ablation_panels` draw the figures.

## Licence

MIT. See LICENSE. The dataset carries its own licence.

## Citation

The paper describing the dataset is under review. A citation will be added here when it is published.
