# TC3-VQA

Code that built the TC3-VQA dataset and ran the evaluations in the accompanying paper. TC3-VQA is a visual
question answering dataset for Tactical Combat Casualty Care. Each item pairs frames from a public TCCC video with a
recognition question, a doctrine question, a clinical-reasoning question and a procedural question, or with a
question the frames cannot answer. Doctrine, reasoning and procedural answers are verbatim passages of public TCCC
documents, cited by character offset.

The dataset is published separately at https://doi.org/10.5281/zenodo.22818562. This repository holds the scripts and prompts
that produced it and the scripts that scored the baseline models.

## Layout

    tc3_vqa/paths.py       working directories, read from environment variables
    tc3_vqa/frame_select   shot detection, frame sampling and filtering, window scoring
    tc3_vqa/corpus         doctrine corpus: sources, download, chunking, embeddings, index
    tc3_vqa/perception     concept assignment and its checks
    tc3_vqa/annotation     equipment boxes, answer regions, caption masks
    tc3_vqa/construction   passage selection, question assembly, regeneration
    tc3_vqa/verification   claim-level entailment gate
    tc3_vqa/consensus      recognition consensus of six models
    tc3_vqa/release        release assembly and the public package
    tc3_vqa/eval           baseline evaluation, controls, release-wide audit
    tc3_vqa/ablation       reference answers versus model-authored answers
    tc3_vqa/analysis       figures and table checks
    tc3_vqa/adjudication   the physician adjudication sample
    prompts/                prompts of the steps run through a vision-language model with file access
    scripts/                run_prompt_batch.py, which replays those prompts through the Messages API
    configs/                SLURM examples for the GPU steps
    docs/pipeline.md        every script of the build, in order

## Installation

Python 3.10 or later. The GPU steps were run with vLLM 0.15 on NVIDIA H200 GPUs. The entailment gate uses
MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli and retrieval uses BAAI/bge-large-en-v1.5.

    pip install -r requirements.txt
    pip install -e .

Set the working directories before running anything (see configs/README.md). Scripts run as modules, for example
`python -m tc3_vqa.frame_select.select_frames --help`.

## Pipeline

The dataset was built in six stages; `docs/pipeline.md` lists every script in the order it was run. Steps that
needed a model to read the frames were run through a vision-language model with file access (Claude Opus 4.8);
those prompts are in `prompts/` and can be replayed with
`python scripts/run_prompt_batch.py prompts/<step>.md manifest.jsonl --out-dir <dir> --frames-dir <frames>`.

1. **Doctrine corpus** (`tc3_vqa.corpus`) downloads the source documents, chunks them at headings, and builds the
   BGE-large embeddings and the FAISS index.
2. **Frames** (`tc3_vqa.frame_select`) splits each video into shots, samples and filters candidate frames, and keeps
   the windows that show an intervention.
3. **Perception** (`tc3_vqa.perception`) assigns one concept per window, re-checks it under a strict prompt, and
   sets review tiers from an independent audit of the frames.
4. **Annotation** (`tc3_vqa.annotation`) proposes, cross-checks and refines the equipment boxes, then adds answer
   regions, body regions and caption masks.
5. **Questions** (`tc3_vqa.construction`, `tc3_vqa.verification`) retrieves the doctrine passages of each scene,
   writes the four question types with every answer matched to its exact corpus span, gates them on claim-level
   entailment, and regenerates whatever the reviews flag.
6. **Release** (`tc3_vqa.consensus`, `tc3_vqa.adjudication`, `tc3_vqa.release`) runs the six-model recognition
   consensus, folds in the rater notes, records source licences and reachability, computes the frame features, and
   writes the public package.

## Evaluation

    python -m tc3_vqa.eval.build_eval_inputs
    sbatch --export=ALL,MODEL=<hf id>,TAG=<tag> configs/baseline_eval.sbatch
    TC3_VQA_RUNS=<runs> python -m tc3_vqa.eval.score_eval <tag>
    python -m tc3_vqa.eval.score_doctrine <runs>/doctrine_<tag>.jsonl      # strict, partial and unsupported-claim rate; reads items and doctrine chunks from TC3_VQA_DATA
    python -m tc3_vqa.eval.build_doctrine_mcq --qtype easy
    python -m tc3_vqa.eval.build_control_inputs && python -m tc3_vqa.eval.score_controls
    python -m tc3_vqa.eval.release_audit --model <hf id> --tag <tag> && python -m tc3_vqa.eval.release_audit_analyze --models <tags>
    python -m tc3_vqa.eval.build_yolo_dataset <out dir> && yolo detect train model=yolov8s.pt data=<out dir>/data.yaml epochs=100 imgsz=640

The comparison with model-authored QA is in tc3_vqa/ablation: `gen_conventional` (open models) and `prep_answer_only`
with `collect_batches` (API-served model) produce the arms, `build_judge_inputs` builds blinded judge inputs,
`judge` runs the local judges, and `analyze`, `figure_data` and `paired_rubric` aggregate them. `gen_conventional`
takes `--concept-given` to state the audited concept in the prompt, or `--concept-from <tag>` to state the concept
the same model chose in the consensus run `<tag>`.

`python -m tc3_vqa.analysis.verify_tables` recomputes every table cell of the paper; `make_plots`,
`ablation_examples` and `ablation_panels` draw the figures.

## Licence

MIT. See LICENSE. The dataset carries its own licence.

## Citation

The paper describing the dataset is under review. A citation will be added here when it is published.
