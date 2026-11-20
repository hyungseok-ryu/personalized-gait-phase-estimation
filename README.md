# Two-Stage Gait-Phase Personalization

## Paper

**Two-Stage Personalized Gait Phase Estimation in Stroke Survivors During Exoskeleton-Assisted Walking: An Offline Feasibility Study**

- Authors: Hyungseok Ryu and Pilwon Hur (corresponding author)
- Journal: *IEEE Transactions on Neural Systems and Rehabilitation Engineering* (accepted for publication, November 2026)
- Preprint: [arXiv:2609.14984](https://doi.org/10.48550/arXiv.2609.14984)
- DOI: [10.1109/TNSRE.2026.3755732](https://doi.org/10.1109/TNSRE.2026.3755732)

If you use this code, the pretrained checkpoints, or the walking dataset, please cite this paper.

## Overview

This directory contains the reference training implementation of the primary method described in the paper:

1. Stage 1 participant-specific calibration using an LwF-regularized objective.
2. Stage 2 prequential cycle-wise refinement using conditional LoRA updates.

The release intentionally excludes LOSO hyperparameter selection, ablation studies, robustness analyses, plotting, and paper-specific result aggregation. Adaptation hyperparameters are supplied through command-line arguments. Values used for a reported participant/model configuration are provided in the paper, not in this code package.

## Scope

This is a training and sequential-adaptation reference implementation. It includes the released IMU/FSR walking dataset, raw-data preprocessing, functional IMU alignment, input window construction and scaling, Stage 1 calibration, and Stage 2 replay. Paper-table reproduction remains outside this package.

The released FSR-derived HS annotations reproduce the phase targets used in the study. An automatic threshold-based FSR detector is also provided for exploratory processing of unannotated trials. FSR-derived labels are used for calibration, Stage 2 triggering, update supervision, and evaluation; the FSR is never used as an estimator input.

## Required inputs

The raw dataset is expected to contain participant folders with walking trials in the following form:

```text
SUB1/
  static/imu_static.csv
  normal_trial_1/imu_thigh_raw.csv
  normal_trial_1/fsr_raw.csv
  ...
```

`preprocess_raw.py` converts each walking-trial directory into:

```text
X_data_window_<WINDOW>.npy
Y_data_window_<WINDOW>.npy
```

`X_data_window_<WINDOW>.npy` contains one row per 100-Hz sample. Its first seven columns are tri-axial acceleration, tri-axial angular velocity, and thigh pitch angle; the final column is the heel-strike indicator. `Y_data_window_<WINDOW>.npy` contains the sine phase, cosine phase, and phase-rate targets and must have `len(X) - WINDOW + 1` rows.

Training also requires:

- A participant static-IMU CSV, included in the raw dataset.
- Reference static and walking CSVs used for functional alignment, included under `configs/reference_data`.
- The seven-channel min/max scaler derived from pre-training data, included as `configs/angle_input_scalers.yaml`.
- A pretrained checkpoint for the selected architecture, included under `pretrained_checkpoints/`.
- The Stage 1 calibration trial and later Stage 2 trials. Their chronological order is defined in the public dataset manifest.

The calibration trial is used for functional alignment and Stage 1 only. It cannot also be supplied as a Stage 2 trial.

## Installation

The code was tested with Python 3.10.12, PyTorch 2.5.1, NumPy 2.2.6, and PyYAML 5.4.1.

```bash
python -m pip install -r requirements.txt
```

## Preprocess the released dataset

Run from the repository root. The following command processes all released participants and creates the arrays used by the training runner:

```bash
python preprocess_raw.py \
  --dataset-root stroke_walking_dataset \
  --output-root processed_public_dataset
```

The default manifest contains affected-side information, recorded condition order, and the FSR-derived HS annotations used to generate the study targets. To redetect HS events directly from the released FSR values instead, add `--detect-events`. Redetection is intended for exploratory use and can differ slightly from the released annotations because the public recordings contain only the retained analysis intervals.

## Train from the released dataset

`run_public_dataset.py` reads `dataset_summary.json`, uses the first trial in the recorded order for Stage 1, and supplies every later trial to Stage 2 in chronological order. Hyperparameters remain explicit command-line inputs.

```bash
python run_public_dataset.py \
  --raw-dataset-root stroke_walking_dataset \
  --processed-root processed_public_dataset \
  --participant SUB1 \
  --model TRANSFORMER \
  --pretrained-checkpoint pretrained_checkpoints/TRANSFORMER_WINDOW_120.pt \
  --lr-stage1 STAGE1_LR \
  --stage1-epochs STAGE1_EPOCHS \
  --lwf-lambda LWF_WEIGHT \
  --lr-stage2 STAGE2_LR \
  --lora-rank LORA_RANK \
  --lora-alpha LORA_ALPHA \
  --gamma TRIGGER_GAMMA \
  --output-dir personalization_output
```

Replace the capitalized hyperparameter placeholders with the desired values. The participant/model configurations used in the paper are reported in the paper rather than hard-coded in this package.

## Train from existing processed arrays

Run from the repository root. Architecture YAML files are included in `configs`; `--model-config` is only needed to override them.

```bash
python run_personalization.py \
  --model TRANSFORMER \
  --pretrained-checkpoint PATH_TO_PRETRAINED_CHECKPOINT \
  --calibration-trial PATH_TO_CALIBRATION_TRIAL \
  --stage2-trials TRIAL_1 TRIAL_2 TRIAL_3 \
  --participant-static PATH_TO_PARTICIPANT_STATIC_CSV \
  --reference-static PATH_TO_REFERENCE_STATIC_CSV \
  --reference-walking PATH_TO_REFERENCE_WALKING_CSV \
  --scaler PATH_TO_SCALER_YAML \
  --affected-side L \
  --lr-stage1 STAGE1_LR \
  --stage1-epochs STAGE1_EPOCHS \
  --lwf-lambda LWF_WEIGHT \
  --lr-stage2 STAGE2_LR \
  --lora-rank LORA_RANK \
  --lora-alpha LORA_ALPHA \
  --gamma TRIGGER_GAMMA \
  --output-dir personalization_output
```

Use `--help` for execution options.

## Implemented protocol

Stage 1 uses all windows fully contained within complete gait cycles of the first walking trial. The pretrained model is copied into a frozen teacher and a trainable student. The student is trained for the supplied fixed epoch count with

```text
L_S1 = L_task + lambda * L_distill
L_task = L_phase + 0.02 * L_rate
```

The sample-level sine-cosine MSE of the calibrated model over the calibration cycles is recorded as `L_ref`. It excludes the phase-rate and distillation terms.

Stage 2 follows test-then-adapt ordering. Each cycle is first predicted with parameters fixed for that cycle. Its sample-level sine-cosine MSE is `L_t`, and an update is triggered when

```text
L_t >= gamma * L_ref
```

The first triggered update uses only the current cycle. Later updates use the current cycle together with the immediately preceding cycle, for two cycles in total. LoRA parameters are updated for one epoch using `L_task`. A snapshot trained from cycle `t` is applied to the inference model at the onset of cycle `t+2`. The adaptation model accumulates successive updates while the inference model receives the delayed snapshots at cycle boundaries.

LoRA is injected into the final fully connected layer of the LSTM, the output linear projection of the TCN, and the self-attention output plus feed-forward projections of the Transformer. All other Stage 1 parameters remain frozen during Stage 2.

## Outputs

The output directory contains:

- `alignment_matrix.txt`: orthonormal participant-to-reference rotation.
- `stage1_model.pt`: calibrated model and basic checkpoint metadata.
- `stage2_latest_trained_model.pt`: latest cumulative LoRA state, including snapshots not yet applied at replay end.
- `stage2_final_inference_model.pt`: state used by the final replayed inference cycle.
- `stage1_history.csv`: Stage 1 optimization history.
- `cycle_metrics.csv`: pre-update loss, phase RMSE, trigger decision, and inference/update IDs per cycle.
- `predictions.npz`: Stage 2 targets and prequential predictions.
- `summary.json`: compact run summary, including the actual trainable parameter count and percentage.
- `run_config.json`: complete command-line configuration.

To reload a Stage 2 checkpoint, instantiate the same base model, inject LoRA with the saved rank and alpha, and then load the checkpoint's `model` state dictionary.

## License

The source code and pretrained model checkpoints are released under the MIT
License. The walking dataset and data-derived configuration files are released
under CC BY 4.0. See `LICENSE` and `DATA_LICENSE` for the exact scope and terms.
