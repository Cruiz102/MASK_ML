import torch
import torch.nn as nn
import torch.nn.functional as F 
from dataclasses import dataclass, Field


@dataclass
class MLPClassificationConfig:
    hidden_size: int
    num_classes: int
    input_size: int
    num_layers: int
    sigmoid_output: bool


class MLPClassification(nn.Module):
    def __init__(
        self,
        cfg: MLPClassificationConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_layers = cfg.num_layers
        h = [cfg.hidden_size] * (cfg.num_layers - 1)
        self.flat = nn.Flatten()
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([cfg.input_size] + h, h + [cfg.num_classes])
        )
        self.sigmoid_output = cfg.sigmoid_output

    def forward(self, x):
        x = self.flat(x)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
