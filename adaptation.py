import copy
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from .lora import inject_lora, trainable_parameters
except ImportError:
    from lora import inject_lora, trainable_parameters


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


@dataclass
class Stage2Result:
    training_model: nn.Module
    inference_model: nn.Module
    rows: list[dict]
    y_true: np.ndarray
    y_pred: np.ndarray
    injected_layers: list[str]
    pending_snapshots: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def task_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    phase = nn.functional.mse_loss(output[:, :2], target[:, :2])
    phase_rate = nn.functional.mse_loss(output[:, 2], target[:, 2])
    return phase + 0.02 * phase_rate


def phase_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(output[:, :2], target[:, :2])


def train_stage1(
    student: nn.Module,
    teacher: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    lwf_lambda: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[dict]:
    set_seed(seed)
    teacher.eval()
    teacher.requires_grad_(False)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    loader = DataLoader(ArrayDataset(x, y), batch_size=batch_size, shuffle=True)
    history = []
    for epoch in range(1, epochs + 1):
        student.train()
        task_total = 0.0
        distillation_total = 0.0
        objective_total = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = student(x_batch)
            with torch.no_grad():
                teacher_output = teacher(x_batch)
            supervised = task_loss(output, y_batch)
            distillation = task_loss(output, teacher_output)
            objective = supervised + lwf_lambda * distillation
            objective.backward()
            optimizer.step()
            task_total += supervised.item()
            distillation_total += distillation.item()
            objective_total += objective.item()
        count = len(loader)
        history.append(
            {
                "epoch": epoch,
                "task_loss": task_total / count,
                "distillation_loss": distillation_total / count,
                "objective": objective_total / count,
            }
        )
    student.eval()
    return history


def evaluate(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    loader = DataLoader(ArrayDataset(x, y), batch_size=batch_size, shuffle=False)
    squared_error_sum = 0.0
    phase_value_count = 0
    true_values = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            output = model(x_batch.to(device))
            target = y_batch.to(device)
            squared_error_sum += nn.functional.mse_loss(
                output[:, :2], target[:, :2], reduction="sum"
            ).item()
            phase_value_count += target[:, :2].numel()
            true_values.append(y_batch.numpy())
            predictions.append(output.cpu().numpy())
    if phase_value_count == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    return (
        squared_error_sum / phase_value_count,
        np.concatenate(true_values),
        np.concatenate(predictions),
    )


def phase_rmse_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_phase = np.mod(np.arctan2(y_true[:, 0], y_true[:, 1]) / (2 * np.pi), 1.0)
    pred_phase = np.mod(np.arctan2(y_pred[:, 0], y_pred[:, 1]) / (2 * np.pi), 1.0)
    circular_error = np.mod(pred_phase - true_phase + 0.5, 1.0) - 0.5
    return float(100.0 * np.sqrt(np.mean(circular_error**2)))


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _train_stage2_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    loader = DataLoader(ArrayDataset(x, y), batch_size=batch_size, shuffle=True)
    model.train()
    total = 0.0
    for x_batch, y_batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = task_loss(model(x_batch.to(device)), y_batch.to(device))
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def run_stage2(
    stage1_model: nn.Module,
    model_name: str,
    named_cycles: list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]],
    *,
    reference_loss: float,
    learning_rate: float,
    lora_rank: int,
    lora_alpha: int,
    gamma: float,
    replay_cycles: int,
    apply_delay_cycles: int,
    batch_size: int,
    device: torch.device,
) -> Stage2Result:
    if apply_delay_cycles < 0:
        raise ValueError("The application delay cannot be negative.")
    training_model = copy.deepcopy(stage1_model)
    injected_layers = inject_lora(training_model, model_name, lora_rank, lora_alpha)
    training_model.to(device)
    inference_model = copy.deepcopy(training_model).to(device).eval()
    optimizer = torch.optim.Adam(trainable_parameters(training_model), lr=learning_rate)
    replay = deque(maxlen=max(replay_cycles, 0))
    pending: list[tuple[int, int, dict[str, torch.Tensor]]] = []
    rows = []
    all_true = []
    all_pred = []
    global_cycle = 0
    trained_update_id = 0
    inference_update_id = 0

    for trial_name, cycles in named_cycles:
        for cycle_in_trial, (x_cycle, y_cycle) in enumerate(cycles, start=1):
            due = [item for item in pending if item[0] <= global_cycle]
            if due:
                _, inference_update_id, state = due[-1]
                inference_model.load_state_dict(state, strict=True)
                inference_model.to(device).eval()
                pending = [item for item in pending if item[0] > global_cycle]

            loss_t, y_true, y_pred = evaluate(
                inference_model, x_cycle, y_cycle, batch_size, device
            )
            triggered = loss_t >= gamma * reference_loss
            update_loss = float("nan")
            if triggered:
                x_parts = [x_cycle] + [item[0] for item in replay]
                y_parts = [y_cycle] + [item[1] for item in replay]
                update_loss = _train_stage2_epoch(
                    training_model,
                    optimizer,
                    np.concatenate(x_parts),
                    np.concatenate(y_parts),
                    batch_size,
                    device,
                )
                trained_update_id += 1
                pending.append(
                    (
                        global_cycle + apply_delay_cycles,
                        trained_update_id,
                        _snapshot(training_model),
                    )
                )

            replay.append((x_cycle, y_cycle))
            rmse = phase_rmse_percent(y_true, y_pred)
            rows.append(
                {
                    "trial": trial_name,
                    "cycle_in_trial": cycle_in_trial,
                    "global_cycle": global_cycle,
                    "num_windows": len(y_true),
                    "phase_mse": loss_t,
                    "rmse_pct_gc": rmse,
                    "trigger_threshold": gamma * reference_loss,
                    "triggered": int(triggered),
                    "inference_update_id": inference_update_id,
                    "trained_update_id": trained_update_id,
                    "update_loss": update_loss,
                }
            )
            all_true.append(y_true)
            all_pred.append(y_pred)
            global_cycle += 1

    if not rows:
        raise ValueError("No complete Stage 2 gait cycle was found in the supplied trials.")
    return Stage2Result(
        training_model,
        inference_model,
        rows,
        np.concatenate(all_true),
        np.concatenate(all_pred),
        injected_layers,
        len(pending),
    )
