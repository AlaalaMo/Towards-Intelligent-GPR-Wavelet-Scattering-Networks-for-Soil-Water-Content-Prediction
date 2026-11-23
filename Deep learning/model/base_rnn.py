# NEW
import math
from typing import Dict, Any
import torch
import torch.nn as nn


def _count_params(module: nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def eda_rnn_summary(model: "Stand_RNN") -> Dict[str, Any]:
    """Return JSON-style EDA summary for architecture decisions."""
    return {
        "who": "Stand_RNN",
        "what": {
            "input_size": model.input_size,
            "in_channels": model.in_channels,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "bidirectional": model.bidirectional,
            "latent_dim": model.latent_dim,
            "n_params_total": _count_params(model),
        },
        "when": "model_init",
        "where": "encoder_module",
        "why": "verify sequence configuration and parameter scale before training",
        "interpretation": "hidden_size and layers set capacity; total params should be ~29k for hidden_size≈96.",
    }


class Stand_RNN(nn.Module):
    """GRU-based encoder for reducing long 1D signals (e.g., GPR traces) to a compact latent vector.

    Overview:
      Tokenizes the input as a length-L sequence with `in_channels` features per timestep, encodes it
      using a GRU, and projects the final hidden state to a low-dimensional latent representation.

    Example:
      Input:  (batch_size, 8192) or (batch_size, 8192, in_channels)
      Output: (batch_size, latent_dim)

    Args:
      in_channels: Features per timestep (1 for scalar samples).
      input_size: Sequence length (e.g., 8192); used for shape validation only.
      latent_dim: Size of the output latent vector.
      hidden_size: GRU hidden width; choose 96 to target ~29k parameters (single layer).
      num_layers: Number of stacked GRU layers.
      dropout_prob: Dropout probability between GRU layers (applies if num_layers > 1).
      bidirectional: If True, use a bidirectional GRU (concatenates directions).
      print_shape: If True, prints intermediate tensor shapes during forward.
      eda_on_init: If True, prints a JSON-style EDA summary at initialization.
    """

    def __init__(
        self,
        in_channels: int = 1,
        input_size: int = 8192,
        latent_dim: int = 4,
        hidden_size: int = 96,
        num_layers: int = 1,
        dropout_prob: float = 0.3,
        bidirectional: bool = False,
        print_shape: bool = False,
        eda_on_init: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.print_shape = print_shape

        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        proj_in = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(proj_in, latent_dim),
            nn.ReLU(),
        )

        if eda_on_init:
            stats = eda_rnn_summary(self)
            print(stats)

    def _coerce_input(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure shape (B, L, C)."""
        if x.dim() == 2:
            # (B, L) -> (B, L, 1)
            return x.unsqueeze(-1)
        if x.dim() == 3:
            return x
        raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, L[, C]) -> (B, latent_dim)."""
        x = self._coerce_input(x)                           # (B, L, C)
        if self.print_shape:
            print(f"input: {x.shape}")

        # GRU encode; take last hidden state from the top layer
        _, h_n = self.gru(x)                                # h_n: (num_layers*D, B, H)
        h_top = h_n[-1] if not self.bidirectional else torch.cat([h_n[-2], h_n[-1]], dim=-1)
        if self.print_shape:
            print(f"h_top: {h_top.shape}")                  # (B, H) or (B, 2H)

        z = self.fc(h_top)                                  # (B, latent_dim)
        if self.print_shape:
            print(f"latent: {z.shape}")
        return z
