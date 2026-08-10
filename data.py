from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class PreparedTrial:
    name: str
    x: np.ndarray
    y: np.ndarray
    heel_strikes: np.ndarray


def _read_csv_numeric(path: str | Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError(f"Expected a finite numeric CSV: {path}")
    return data


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-8:
        raise ValueError("Cannot construct an IMU alignment axis from a zero vector.")
    return vector / norm


def _principal_axis(data: np.ndarray) -> np.ndarray:
    centered = data - data.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / len(centered)
    values, vectors = np.linalg.eigh(covariance)
    return vectors[:, np.argmax(values)]


def compute_alignment_matrix(
    reference_static_path: str | Path,
    reference_walking_path: str | Path,
    participant_static_path: str | Path,
    calibration_walking: np.ndarray,
    affected_side: str,
) -> np.ndarray:
    side = affected_side.upper()
    if side not in {"L", "R"}:
        raise ValueError("affected_side must be either 'L' or 'R'.")
    reference_static = _read_csv_numeric(reference_static_path)
    reference_walking = _read_csv_numeric(reference_walking_path)
    participant_static = _read_csv_numeric(participant_static_path)

    gravity_reference = _normalize(reference_static[:, :3].mean(axis=0))
    gravity_participant = _normalize(participant_static[:, :3].mean(axis=0))
    sagittal_reference = _principal_axis(reference_walking[:, 3:6])
    sagittal_participant = _principal_axis(calibration_walking[:, 3:6])

    if side == "L" and sagittal_participant[-1] < 0:
        sagittal_participant *= -1
    elif side == "R" and sagittal_participant[-1] > 0:
        sagittal_participant *= -1

    def basis(gravity: np.ndarray, sagittal: np.ndarray) -> np.ndarray:
        x_axis = _normalize(gravity)
        z_axis = _normalize(np.cross(x_axis, sagittal))
        y_axis = _normalize(np.cross(z_axis, x_axis))
        return np.column_stack([x_axis, y_axis, z_axis])

    rotation = basis(gravity_reference, sagittal_reference) @ basis(
        gravity_participant, sagittal_participant
    ).T
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise RuntimeError("The functional alignment matrix is not orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise RuntimeError("The functional alignment matrix is not a proper rotation.")
    return rotation


def load_raw_trial(trial_path: str | Path, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    trial_path = Path(trial_path)
    x_path = trial_path / f"X_data_window_{window_size}.npy"
    y_path = trial_path / f"Y_data_window_{window_size}.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing processed trial arrays in {trial_path}")
    x = np.load(x_path).astype(np.float64, copy=True)
    y = np.load(y_path).astype(np.float64, copy=True)
    if x.ndim != 2 or x.shape[1] < 8:
        raise ValueError(f"Expected X with at least 8 columns in {x_path}; found {x.shape}.")
    expected_targets = len(x) - window_size + 1
    if y.ndim != 2 or y.shape[1] < 3 or len(y) != expected_targets:
        raise ValueError(
            f"Expected Y shape ({expected_targets}, 3+) for {trial_path}; found {y.shape}."
        )
    return x, y[:, :3]


def prepare_trial(
    trial_path: str | Path,
    window_size: int,
    rotation: np.ndarray,
) -> PreparedTrial:
    x, y = load_raw_trial(trial_path, window_size)
    x[:, :3] = x[:, :3] @ rotation.T
    x[:, 3:6] = x[:, 3:6] @ rotation.T
    return PreparedTrial(Path(trial_path).name, x, y, np.flatnonzero(x[:, -1] == 1.0))


def load_scaler(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as stream:
        entries = yaml.safe_load(stream)["scalers"]
    minimum = np.asarray([entry["min"] for entry in entries], dtype=np.float64)
    maximum = np.asarray([entry["max"] for entry in entries], dtype=np.float64)
    if np.any(maximum <= minimum):
        raise ValueError("Every scaler maximum must be greater than its minimum.")
    return minimum, maximum


def _make_windows(
    trial: PreparedTrial,
    window_size: int,
    start: int,
    stop: int,
    scaler: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stop - start < window_size:
        raise ValueError(
            f"Interval [{start}, {stop}) in {trial.name} is shorter than the window size."
        )
    starts = np.arange(start, stop - window_size + 1, dtype=int)
    windows = np.stack([trial.x[index : index + window_size, :-1] for index in starts])
    targets = trial.y[starts]
    heel_strikes = trial.x[starts + window_size - 1, -1]
    minimum, maximum = scaler
    if windows.shape[2] > len(minimum):
        raise ValueError("The scaler does not contain enough input channels.")
    windows = (windows - minimum[: windows.shape[2]]) / (
        maximum[: windows.shape[2]] - minimum[: windows.shape[2]]
    )
    return (
        np.transpose(windows, (0, 2, 1)).astype(np.float32),
        targets.astype(np.float32),
        heel_strikes.astype(np.float32),
    )


def calibration_data(
    trial: PreparedTrial,
    window_size: int,
    scaler: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if len(trial.heel_strikes) < 2:
        raise ValueError(f"No complete gait cycle was found in {trial.name}.")
    x, y, _ = _make_windows(
        trial,
        window_size,
        int(trial.heel_strikes[0]),
        int(trial.heel_strikes[-1]),
        scaler,
    )
    return x, y


def stage2_cycles(
    trial: PreparedTrial,
    window_size: int,
    scaler: tuple[np.ndarray, np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    x, y, heel_strikes = _make_windows(trial, window_size, 0, len(trial.x), scaler)
    indices = np.flatnonzero(heel_strikes == 1.0)
    return [(x[a:b], y[a:b]) for a, b in zip(indices[:-1], indices[1:]) if b > a]
