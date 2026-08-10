#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .adaptation import (
        evaluate,
        phase_rmse_percent,
        run_stage2,
        set_seed,
        train_stage1,
    )
    from .data import (
        calibration_data,
        compute_alignment_matrix,
        load_raw_trial,
        load_scaler,
        prepare_trial,
        stage2_cycles,
    )
    from .models import build_model
except ImportError:
    from adaptation import (
        evaluate,
        phase_rmse_percent,
        run_stage2,
        set_seed,
        train_stage1,
    )
    from data import (
        calibration_data,
        compute_alignment_matrix,
        load_raw_trial,
        load_scaler,
        prepare_trial,
        stage2_cycles,
    )
    from models import build_model


PREVIOUS_REPLAY_CYCLES = 1
UPDATE_APPLY_DELAY_CYCLES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 1 LwF calibration and Stage 2 conditional LoRA adaptation."
    )
    parser.add_argument("--model", required=True, choices=["LSTM", "TCN", "TRANSFORMER"])
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Architecture YAML. Defaults to configs/<MODEL>.yaml in this release.",
    )
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-trial", type=Path, required=True)
    parser.add_argument("--stage2-trials", type=Path, nargs="+", required=True)
    parser.add_argument("--participant-static", type=Path, required=True)
    parser.add_argument("--reference-static", type=Path, required=True)
    parser.add_argument("--reference-walking", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--affected-side", required=True, choices=["L", "R"])

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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow known output files in a non-empty output directory to be replaced.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if name == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available.")
    return device


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def run(args: argparse.Namespace) -> None:
    positive_values = {
        "--lr-stage1": args.lr_stage1,
        "--stage1-epochs": args.stage1_epochs,
        "--lr-stage2": args.lr_stage2,
        "--lora-rank": args.lora_rank,
        "--lora-alpha": args.lora_alpha,
        "--gamma": args.gamma,
        "--window-size": args.window_size,
        "--batch-size": args.batch_size,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.lwf_lambda < 0:
        raise ValueError("--lwf-lambda cannot be negative.")
    if args.model_config is None:
        args.model_config = Path(__file__).resolve().parent / "configs" / f"{args.model}.yaml"

    calibration_path = args.calibration_trial.resolve()
    if any(path.resolve() == calibration_path for path in args.stage2_trials):
        raise ValueError("The calibration trial cannot also be supplied as a Stage 2 trial.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use --overwrite to replace outputs."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    set_seed(args.seed)

    calibration_x_raw, _ = load_raw_trial(args.calibration_trial, args.window_size)
    rotation = compute_alignment_matrix(
        args.reference_static,
        args.reference_walking,
        args.participant_static,
        calibration_x_raw[:, :6],
        args.affected_side,
    )
    np.savetxt(args.output_dir / "alignment_matrix.txt", rotation)
    scaler = load_scaler(args.scaler)
    calibration_trial = prepare_trial(args.calibration_trial, args.window_size, rotation)
    calibration_x, calibration_y = calibration_data(
        calibration_trial, args.window_size, scaler
    )

    teacher = build_model(
        args.model, args.model_config, args.pretrained_checkpoint, device
    )
    student = build_model(
        args.model, args.model_config, args.pretrained_checkpoint, device
    )
    history = train_stage1(
        student,
        teacher,
        calibration_x,
        calibration_y,
        learning_rate=args.lr_stage1,
        epochs=args.stage1_epochs,
        lwf_lambda=args.lwf_lambda,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    reference_loss, calibration_true, calibration_pred = evaluate(
        student, calibration_x, calibration_y, args.batch_size, device
    )
    torch.save(
        {
            "model": student.state_dict(),
            "model_name": args.model,
            "stage": 1,
            "window_size": args.window_size,
        },
        args.output_dir / "stage1_model.pt",
    )
    write_csv(args.output_dir / "stage1_history.csv", history)

    named_cycles = []
    for path in args.stage2_trials:
        trial = prepare_trial(path, args.window_size, rotation)
        named_cycles.append((trial.name, stage2_cycles(trial, args.window_size, scaler)))
    result = run_stage2(
        student,
        args.model,
        named_cycles,
        reference_loss=reference_loss,
        learning_rate=args.lr_stage2,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        gamma=args.gamma,
        replay_cycles=PREVIOUS_REPLAY_CYCLES,
        apply_delay_cycles=UPDATE_APPLY_DELAY_CYCLES,
        batch_size=args.batch_size,
        device=device,
    )
    torch.save(
        {
            "model": result.training_model.state_dict(),
            "model_name": args.model,
            "stage": 2,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "state_role": "latest_trained",
        },
        args.output_dir / "stage2_latest_trained_model.pt",
    )
    torch.save(
        {
            "model": result.inference_model.state_dict(),
            "model_name": args.model,
            "stage": 2,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "state_role": "final_inference",
        },
        args.output_dir / "stage2_final_inference_model.pt",
    )
    write_csv(args.output_dir / "cycle_metrics.csv", result.rows)
    np.savez_compressed(
        args.output_dir / "predictions.npz", y_true=result.y_true, y_pred=result.y_pred
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in result.training_model.parameters()
        if parameter.requires_grad
    )
    all_parameters = sum(parameter.numel() for parameter in result.training_model.parameters())
    summary = {
        "model": args.model,
        "device": str(device),
        "calibration_windows": len(calibration_y),
        "calibration_rmse_pct_gc": phase_rmse_percent(calibration_true, calibration_pred),
        "reference_phase_mse": reference_loss,
        "stage2_cycles": len(result.rows),
        "stage2_mean_cycle_rmse_pct_gc": float(
            np.mean([row["rmse_pct_gc"] for row in result.rows])
        ),
        "stage2_triggered_updates": int(sum(row["triggered"] for row in result.rows)),
        "pending_updates_at_end": result.pending_snapshots,
        "previous_replay_cycles": PREVIOUS_REPLAY_CYCLES,
        "update_apply_delay_cycles": UPDATE_APPLY_DELAY_CYCLES,
        "lora_layers": result.injected_layers,
        "stage2_trainable_parameters": trainable_parameters,
        "stage2_all_parameters": all_parameters,
        "stage2_trainable_percent": 100.0 * trainable_parameters / all_parameters,
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as stream:
        json.dump({key: json_ready(value) for key, value in vars(args).items()}, stream, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to {args.output_dir.resolve()}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
