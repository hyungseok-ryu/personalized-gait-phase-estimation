import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.nn.utils.parametrizations import weight_norm


class LSTMModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = int(config["hidden_size"])
        self.num_layers = int(config["num_layers"])
        self.lstm = nn.LSTM(
            input_size=int(config["input_size"]),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(config["dropout"]),
        )
        self.fc = nn.Linear(self.hidden_size, int(config.get("num_classes", 3)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self.relu = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TCNModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        channels = list(config["tcn_channels"])
        kernel_size = int(config["kernel_size"])
        layers = []
        for index, out_channels in enumerate(channels):
            in_channels = int(config["input_size"]) if index == 0 else channels[index - 1]
            dilation = (
                int(config.get("const_dilation", 1))
                if config.get("dilation_mode", "exp") == "const"
                else 2**index
            )
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    1,
                    dilation,
                    padding,
                    float(config["dropout"]),
                )
            )
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], int(config.get("num_classes", 3)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.network(x)[:, :, -1])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class PredictionHead(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        d_model = int(config["d_model"])
        self.proj = nn.Conv1d(int(config["input_channels"]), d_model, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(config["nhead"]),
            dim_feedforward=int(config["dim_feedforward"]),
            dropout=float(config["trans_dropout"]),
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(config["num_encoder_layers"])
        )
        if int(config["num_decoder_layers"]) > 0:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=int(config["nhead"]),
                dim_feedforward=int(config["dim_feedforward"]),
                dropout=float(config["trans_dropout"]),
                batch_first=True,
                norm_first=False,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=int(config["num_decoder_layers"])
            )
            self.query = nn.Parameter(torch.randn(1, 1, d_model))
        else:
            self.decoder = None
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model, int(config["max_len"]))
        self.att_pool = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1)
        )
        self.head = PredictionHead(
            d_model,
            int(config["head_hidden"]),
            float(config["head_dropout"]),
            int(config["out_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        source = self.pos_enc(self.proj(x).transpose(1, 2))
        memory = self.encoder(source)
        if self.decoder is not None:
            target = self.query.expand(batch_size, 1, -1)
            return self.head(self.decoder(target, memory).squeeze(1))
        scores = self.att_pool(memory).squeeze(-1)
        attention = torch.softmax(scores, dim=1).unsqueeze(-1)
        return self.head((memory * attention).sum(dim=1))


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def build_model(
    model_name: str,
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    constructors = {
        "LSTM": LSTMModel,
        "TCN": TCNModel,
        "TRANSFORMER": TransformerModel,
    }
    model = constructors[model_name.upper()](config)
    model.load_state_dict(load_checkpoint(checkpoint_path, device), strict=True)
    return model.to(device)
