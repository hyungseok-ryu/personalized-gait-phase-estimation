#!/usr/bin/env python3
"""Run personalization from arrays created from the released raw dataset."""

import argparse
import json
from pathlib import Path

try:
    from .run_personalization import run
except ImportError:
    from run_personalization import run


def parse_args() -> argparse.Namespace:
    release_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train Stage 1 and replay Stage 2 for one released participant."
    )
    parser.add_argument("--raw-dataset-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--model", required=True, choices=["LSTM", "TCN", "TRANSFORMER"])
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument(
        "--scaler",
        type=Path,
        default=release_root / "configs" / "angle_input_scalers.yaml",
    )
    parser.add_argument(
        "--reference-static",
        type=Path,
        default=release_root / "configs" / "reference_data" / "reference_imu_data_static.csv",
    )
    parser.add_argument(
        "--reference-walking",
        type=Path,
        default=release_root / "configs" / "reference_data" / "reference_imu_data_walking.csv",
    )
    parser.add_argument("--lr-stage1", type=float, required=True)
    parser.add_argument("--stage1-epochs", type=int, required=True)
    parser.add_argument("--lwf-lambda", type=float, required=True)
    parser.add_argument("--lr-stage2", type=float, required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--lora-alpha", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", type=Path, default=Path("personalization_output"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    public_args = parse_args()
    summary_path = public_args.processed_root / "dataset_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Preprocessing summary not found: {summary_path}. Run preprocess_raw.py first."
        )
    with open(summary_path, "r", encoding="utf-8") as stream:
        summary = json.load(stream)
    if public_args.participant not in summary["participants"]:
        raise KeyError(f"Participant was not preprocessed: {public_args.participant}")
    participant = summary["participants"][public_args.participant]
    processed_participant = public_args.processed_root / public_args.participant
    static_path = (
        public_args.raw_dataset_root
        / public_args.participant
        / "static"
        / "imu_static.csv"
    )

    training_args = argparse.Namespace(
        model=public_args.model,
        model_config=public_args.model_config,
        pretrained_checkpoint=public_args.pretrained_checkpoint,
        calibration_trial=processed_participant / participant["calibration_trial"],
        stage2_trials=[
            processed_participant / name for name in participant["stage2_trials"]
        ],
        participant_static=static_path,
        reference_static=public_args.reference_static,
        reference_walking=public_args.reference_walking,
        scaler=public_args.scaler,
        affected_side=participant["affected_side"],
        lr_stage1=public_args.lr_stage1,
        stage1_epochs=public_args.stage1_epochs,
        lwf_lambda=public_args.lwf_lambda,
        lr_stage2=public_args.lr_stage2,
        lora_rank=public_args.lora_rank,
        lora_alpha=public_args.lora_alpha,
        gamma=public_args.gamma,
        window_size=public_args.window_size,
        batch_size=public_args.batch_size,
        seed=public_args.seed,
        device=public_args.device,
        output_dir=public_args.output_dir,
        overwrite=public_args.overwrite,
    )
    run(training_args)


if __name__ == "__main__":
    main()
