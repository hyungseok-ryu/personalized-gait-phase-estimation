#!/usr/bin/env python3
"""Convert the released IMU/FSR CSV files into Stage 1/2 training arrays."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml


REQUIRED_IMU_COLUMNS = [
    "timestamp",
    "angle",
    "linear_acceleration_x",
    "linear_acceleration_y",
    "linear_acceleration_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
]
REQUIRED_FSR_COLUMNS = ["timestamp", "data"]


def parse_args() -> argparse.Namespace:
    release_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create window-aligned gait-phase targets from the released IMU/FSR CSV files."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--participants", nargs="+", default=None)
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=release_root / "configs" / "public_dataset_manifest.yaml",
        help="Participant metadata and FSR-derived HS annotations for the released dataset.",
    )
    parser.add_argument(
        "--detect-events",
        action="store_true",
        help="Ignore released HS annotations and redetect events from the FSR signal.",
    )
    parser.add_argument("--fsr-threshold-ratio", type=float, default=0.2)
    parser.add_argument("--minimum-cycle-ms", type=float, default=1000.0)
    parser.add_argument("--sampling-rate", type=float, default=100.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_named_csv(path: Path, required: list[str]) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding="utf-8")
    if table.ndim == 0:
        table = table.reshape(1)
    names = set(table.dtype.names or ())
    missing = set(required) - names
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    result = {name: np.asarray(table[name], dtype=np.float64) for name in required}
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError(f"Non-finite values found in {path}")
    return result


def nearest_values(
    target_timestamps: np.ndarray,
    source_timestamps: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    order = np.argsort(source_timestamps)
    source_timestamps = source_timestamps[order]
    source_values = source_values[order]
    right = np.searchsorted(source_timestamps, target_timestamps, side="left")
    right = np.clip(right, 0, len(source_timestamps) - 1)
    left = np.clip(right - 1, 0, len(source_timestamps) - 1)
    choose_left = np.abs(target_timestamps - source_timestamps[left]) <= np.abs(
        source_timestamps[right] - target_timestamps
    )
    indices = np.where(choose_left, left, right)
    return source_values[indices]


def detect_heel_strikes(
    imu_timestamps: np.ndarray,
    fsr_timestamps: np.ndarray,
    fsr_values: np.ndarray,
    *,
    window_size: int,
    threshold_ratio: float,
    minimum_interval: int,
) -> list[int]:
    if not 0.0 < threshold_ratio < 1.0:
        raise ValueError("--fsr-threshold-ratio must be between zero and one.")
    synchronized = nearest_values(imu_timestamps, fsr_timestamps, fsr_values)
    threshold = synchronized.min() + threshold_ratio * (
        synchronized.max() - synchronized.min()
    )
    above = synchronized > threshold
    candidates = (np.flatnonzero((~above[:-1]) & above[1:]) + 1).tolist()
    retained: list[int] = []
    for index in candidates:
        if not retained or index - retained[-1] >= minimum_interval:
            retained.append(int(index))

    # Released trials begin one input window before the first retained HS and
    # end at the final retained HS. Re-anchor threshold crossings to those
    # boundaries to avoid treating a transition at the cropped edge as an HS.
    retained = [index for index in retained if index >= window_size]
    if not retained:
        raise ValueError("No FSR heel-strike transition was detected.")
    offset = window_size - retained[0]
    heel_strikes = [index + offset for index in retained]
    heel_strikes = [index for index in heel_strikes if 0 <= index < len(imu_timestamps)]
    final_index = len(imu_timestamps) - 1
    if final_index - heel_strikes[-1] < minimum_interval / 2:
        heel_strikes[-1] = final_index
    else:
        heel_strikes.append(final_index)
    heel_strikes[0] = window_size
    return heel_strikes


def validate_heel_strikes(indices: list[int], sample_count: int, window_size: int) -> None:
    if len(indices) < 2:
        raise ValueError("At least two heel strikes are required.")
    if indices != sorted(set(indices)):
        raise ValueError("Heel-strike indices must be strictly increasing.")
    if indices[0] < window_size - 1 or indices[-1] >= sample_count:
        raise ValueError(
            f"Heel-strike indices fall outside the usable sample range: {indices}"
        )


def make_phase_targets(
    sample_count: int,
    heel_strikes: list[int],
    window_size: int,
    sampling_rate: float,
) -> np.ndarray:
    phase = np.full(sample_count, np.nan, dtype=np.float64)
    phase_rate = np.full(sample_count, np.nan, dtype=np.float64)
    for left, right in zip(heel_strikes[:-1], heel_strikes[1:]):
        start = left - 1
        length = right - start
        phase[start:right] = np.linspace(0.0, 1.0, length, endpoint=False)
        phase_rate[start:right] = sampling_rate / length
    phase[-1] = 0.0
    phase_rate[-1] = phase_rate[-2]
    phase = phase[window_size - 1 :]
    phase_rate = phase_rate[window_size - 1 :]
    if not np.isfinite(phase).all() or not np.isfinite(phase_rate).all():
        raise ValueError("The retained heel strikes do not cover the target interval.")
    return np.column_stack(
        [np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase), phase_rate]
    )


def trial_sort_key(name: str, condition_order: list[str]) -> tuple[int, int]:
    condition = next((item for item in condition_order if name.startswith(f"{item}_")), None)
    if condition is None:
        return len(condition_order), 10**6
    match = re.search(r"_trial_(\d+)$", name)
    trial = int(match.group(1)) if match else 10**6
    return condition_order.index(condition), trial


def process_trial(
    trial_dir: Path,
    output_dir: Path,
    *,
    affected_side: str,
    annotated_heel_strikes: list[int] | None,
    args: argparse.Namespace,
) -> dict:
    imu = read_named_csv(trial_dir / "imu_thigh_raw.csv", REQUIRED_IMU_COLUMNS)
    fsr = read_named_csv(trial_dir / "fsr_raw.csv", REQUIRED_FSR_COLUMNS)
    order = np.argsort(imu["timestamp"])
    imu = {name: values[order] for name, values in imu.items()}
    sample_count = len(imu["timestamp"])

    if annotated_heel_strikes is not None and not args.detect_events:
        heel_strikes = [int(index) for index in annotated_heel_strikes]
        event_source = "released_fsr_annotation"
    else:
        minimum_interval = int(round(args.minimum_cycle_ms * args.sampling_rate / 1000.0))
        heel_strikes = detect_heel_strikes(
            imu["timestamp"],
            fsr["timestamp"],
            fsr["data"],
            window_size=args.window_size,
            threshold_ratio=args.fsr_threshold_ratio,
            minimum_interval=minimum_interval,
        )
        event_source = "automatic_fsr_detection"
    validate_heel_strikes(heel_strikes, sample_count, args.window_size)

    angle = imu["angle"].copy()
    if affected_side.upper() == "L":
        angle *= -1.0
    heel_indicator = np.zeros(sample_count, dtype=np.float64)
    heel_indicator[heel_strikes] = 1.0
    x = np.column_stack(
        [
            imu["linear_acceleration_x"],
            imu["linear_acceleration_y"],
            imu["linear_acceleration_z"],
            imu["angular_velocity_x"],
            imu["angular_velocity_y"],
            imu["angular_velocity_z"],
            angle,
            heel_indicator,
        ]
    )
    y = make_phase_targets(sample_count, heel_strikes, args.window_size, args.sampling_rate)
    if len(y) != sample_count - args.window_size + 1:
        raise RuntimeError("Window-aligned target length is inconsistent with the input length.")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"X_data_window_{args.window_size}.npy", x)
    np.save(output_dir / f"Y_data_window_{args.window_size}.npy", y)
    metadata = {
        "source_trial": str(trial_dir.resolve()),
        "affected_side": affected_side.upper(),
        "samples": sample_count,
        "heel_strikes": heel_strikes,
        "complete_cycles": len(heel_strikes) - 1,
        "event_source": event_source,
        "window_size": args.window_size,
        "sampling_rate_hz": args.sampling_rate,
    }
    with open(output_dir / "preprocessing_metadata.json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return metadata


def main() -> None:
    args = parse_args()
    if args.window_size <= 0 or args.sampling_rate <= 0 or args.minimum_cycle_ms <= 0:
        raise ValueError("Window size, sampling rate, and minimum cycle duration must be positive.")
    with open(args.manifest, "r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not args.detect_events and int(manifest["window_size"]) != args.window_size:
        raise ValueError(
            "Released HS annotations are defined for window size "
            f"{manifest['window_size']}; use that value or pass --detect-events."
        )

    participants = args.participants or sorted(manifest["participants"])
    summary = {"dataset_root": str(args.dataset_root.resolve()), "participants": {}}
    for participant in participants:
        if participant not in manifest["participants"]:
            raise KeyError(f"Participant is not defined in the manifest: {participant}")
        participant_root = args.dataset_root / participant
        if not participant_root.is_dir():
            raise FileNotFoundError(participant_root)
        info = manifest["participants"][participant]
        trial_dirs = sorted(
            [path for path in participant_root.iterdir() if path.is_dir() and "_trial_" in path.name],
            key=lambda path: trial_sort_key(path.name, info["condition_order"]),
        )
        participant_output = args.output_root / participant
        if participant_output.exists() and any(participant_output.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {participant_output}. Use --overwrite."
            )
        trial_rows = []
        for trial_dir in trial_dirs:
            annotation = info.get("heel_strikes", {}).get(trial_dir.name)
            metadata = process_trial(
                trial_dir,
                participant_output / trial_dir.name,
                affected_side=info["affected_side"],
                annotated_heel_strikes=annotation,
                args=args,
            )
            trial_rows.append({"trial": trial_dir.name, **metadata})
        if not trial_rows:
            raise ValueError(f"No walking trials found for {participant}")
        summary["participants"][participant] = {
            "affected_side": info["affected_side"],
            "condition_order": info["condition_order"],
            "calibration_trial": trial_rows[0]["trial"],
            "stage2_trials": [row["trial"] for row in trial_rows[1:]],
            "trials": trial_rows,
        }
        print(
            f"Processed {participant}: calibration={trial_rows[0]['trial']}, "
            f"Stage 2 trials={len(trial_rows) - 1}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    with open(args.output_root / "dataset_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(f"Saved processed arrays to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
