# model/layers.py
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    """
    PINN backbone MLP.

    in_dim = 2 (z, t)
    out_dim = 1 (T)
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden_dim: int = 128,
        num_hidden_layers: int = 4,
        activation: str = "tanh",
    ) -> None:
        super().__init__()

        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

        layers: List[nn.Module] = []
        act = get_activation(activation)

        # input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act)

        # hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)

        # output layer
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
