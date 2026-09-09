README.md
=========

# Sterile Package Grounding

Code accompanying the RA-L work on grounding egocentric sterile-package-opening demonstrations into robot-relevant manipulation phases.

The pipeline follows:

**Observe → Ground → Translate → Execute**

Human demonstrations provide task semantics and phase ordering. Robot execution is determined by robot embodiment, sensing configuration, contact strategy, and a manually designed semantic-to-skill mapping.

## Method overview

The grounding pipeline combines:

- hand geometry
- optical-flow-derived motion cues
- Pixtral-12B semantic reasoning

The full method is hybrid:

**geometry + motion proposals → Pixtral semantic resolution → grounded robot-relevant phases**

The main robot-relevant phases are:

- `locate_flaps`
- `grasp_flaps`
- `peel_apart`
- `drop_contents`

The mapping from grounded semantic phases to ABB YuMi robot skills is manually designed. This repository therefore does not claim end-to-end video-to-robot policy learning.

## Repository structure

```text
sterile-package-grounding/
├── grounding/
│   ├── build_sequences.py
│   ├── detect_bt_events.py
│   ├── extract_bt_features.py
│   ├── extract_hand_geometry.py
│   ├── finalize_bt_labels.py
│   ├── pixtral_detect_start_phase.py
│   ├── pixtral_labeler.py
│   ├── pixtral_refine_boundaries.py
│   └── vocabulary.py
├── evaluation/
│   ├── evaluate_ablation_geometry_motion.py
│   ├── evaluate_ablation_pixtral_only.py
│   ├── evaluate_cras_grounding.py
│   └── evaluate_cras_tiou.py
├── scripts/
│   ├── run_bt_pipeline.py
│   └── run_pixtral_only_ablation.py
├── reproducibility/
│   ├── environment.txt
│   ├── pipeline_sha256.txt
│   └── README.md
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Grounding conditions

The repository supports three evaluation conditions:

- **Geometry + motion only**
- **Pixtral only**
- **Full hybrid: geometry + motion + Pixtral**

The hybrid method uses low-level geometry and motion cues to generate candidate manipulation events, then applies Pixtral to resolve semantic ambiguity.

The Pixtral-only condition is retained as an ablation baseline and uses RGB observations without geometry or optical-flow-derived event proposals.

## Dataset

The study uses egocentric RGB-D demonstrations of sterile package opening.

- 77 demonstrations were collected.
- 76 sequences were available for automatic processing.
- One source sequence was unavailable in the transferred processing set.
- An independent manually annotated subset was used for grounding evaluation.

Raw videos, RGB-D frames, participant data, and model weights are not included in this repository.

## Running the grounding pipeline

```bash
python scripts/run_bt_pipeline.py \
  --dataset /path/to/dataset \
  --workdir /path/to/output
```

To process a selected sequence:

```bash
python scripts/run_bt_pipeline.py \
  --dataset /path/to/dataset \
  --workdir /path/to/output \
  --sequence P01_SEQ_0001
```

Use `--force` to rerun stages even when outputs already exist.

## Pixtral-only ablation

```bash
python scripts/run_pixtral_only_ablation.py \
  --batch_size 1 \
  --temporal_window 5
```

The standalone Pixtral baseline uses RGB-only framewise visual-state assessment followed by temporally persistent state-transition detection.

Model:

```text
mistral-community/pixtral-12b
```

## Evaluation

Grounding performance is evaluated using:

- boundary absolute error
- median absolute error
- thresholded boundary accuracy
- temporal IoU

Evaluation scripts are located in `evaluation/`.

## Reproducibility

The recorded processing environment is documented in:

```text
reproducibility/environment.txt
```

SHA256 checksums of the principal publication-pipeline source files are recorded in:

```text
reproducibility/pipeline_sha256.txt
```

The core publication algorithm files were re-hashed before creation of this repository and matched the recorded SHA256 values.

The public `scripts/run_bt_pipeline.py` launcher was subsequently made portable by removing KCL cluster-specific absolute paths and adding configurable dataset and working-directory arguments. This portability change does not alter the grounding algorithms used to produce the reported results.

## Software environment

The recorded environment included:

- Python 3.11.15
- PyTorch 2.13.0+cu130
- CUDA 13.0
- Transformers 5.14.1
- pandas 3.0.5
- NumPy 2.3.5
- MediaPipe 1.0.1
- NVIDIA L40S

Exact recorded details are listed in `reproducibility/environment.txt`.

## Data and model availability

This repository does not include:

- raw participant recordings
- extracted image frames
- RGB-D data
- model weights
- generated intermediate features
- cluster-specific logs

Model weights should be obtained separately from the corresponding model provider.

## Citation

Citation information will be added following publication.

## Licence

See `LICENSE`.
