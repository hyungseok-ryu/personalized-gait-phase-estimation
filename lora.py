import math
from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int, alpha: int):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive.")
        original_layer.requires_grad_(False)
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_layer(x) + self.lora_B(self.lora_A(x)) * self.scaling

    @property
    def weight(self) -> torch.Tensor:
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        return self.original_layer.weight + delta.to(self.original_layer.weight.dtype)

    @property
    def bias(self) -> torch.Tensor | None:
        return self.original_layer.bias


def _inject(
    module: nn.Module,
    rank: int,
    alpha: int,
    target_names: list[str],
    parent_path: str = "",
) -> list[str]:
    injected = []
    for name, child in list(module.named_children()):
        path = f"{parent_path}.{name}" if parent_path else name
        if isinstance(child, nn.Linear) and any(target in path for target in target_names):
            setattr(module, name, LoRALinear(child, rank, alpha))
            injected.append(path)
        else:
            injected.extend(_inject(child, rank, alpha, target_names, path))
    return injected


def inject_lora(model: nn.Module, model_name: str, rank: int, alpha: int) -> list[str]:
    targets = {
        "LSTM": ["fc", "linear", "regressor"],
        "TCN": ["conv1", "conv2", "downsample", "linear", "fc"],
        "TRANSFORMER": ["out_proj", "linear1", "linear2"],
    }[model_name.upper()]
    model.requires_grad_(False)
    injected = _inject(model, rank, alpha, targets)
    if not injected:
        raise RuntimeError(f"No LoRA target layers were found for {model_name}.")
    return injected


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)
