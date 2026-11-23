# NEW
import torch
import torch.nn as nn
from typing import Dict, Any


def _count_params(module: nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def eda_transformer_summary(model: "Stand_Transformer") -> Dict[str, Any]:
    """Return JSON-style summary of the Transformer model configuration."""
    return {
        "who": "Stand_Transformer",
        "what": {
            "input_size": model.input_size,
            "patch_size": model.patch_size,
            "num_patches": model.num_patches,
            "d_model": model.d_model,
            "nhead": model.nhead,
            "num_layers": model.num_layers,
            "dim_feedforward": model.dim_feedforward,
            "latent_dim": model.latent_dim,
            "n_params_total": _count_params(model),
        },
        "when": "model_init",
        "where": "encoder_module",
        "why": "verify model complexity, tokenization granularity, and parameter scale",
        "interpretation": (
            "Patch embedding and Transformer depth determine receptive field and capacity; "
            "parameter count should be consistent with low-data regime (~80k–120k for d_model=64)."
        ),
    }


class Stand_Transformer(nn.Module):
    """Transformer-based encoder for reducing long 1D signals (e.g., GPR traces) into a compact latent vector.

    Overview:
      Splits the input into fixed-size patches, embeds each patch into a latent dimension,
      adds positional encodings, and passes the resulting sequence through a Transformer
      encoder. The pooled output represents the global latent embedding.

    Example:
      Input:  (batch_size, 8192)
      Output: (batch_size, latent_dim)

    Args:
      input_size: Length of input signal (e.g., 8192).
      patch_size: Number of samples per token (defines number of patches).
      d_model: Transformer embedding dimension.
      nhead: Number of attention heads.
      num_layers: Number of Transformer encoder layers.
      dim_feedforward: Hidden size of feedforward sublayer.
      latent_dim: Dimension of final latent vector.
      dropout_prob: Dropout probability.
      print_shape: If True, print tensor shapes during forward.
      eda_on_init: If True, print model EDA summary on initialization.
    """

    def __init__(
        self,
        input_size: int = 8192,
        patch_size: int = 256,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        latent_dim: int = 4,
        dropout_prob: float = 0.4,
        print_shape: bool = False,
        eda_on_init: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = input_size // patch_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.latent_dim = latent_dim
        self.print_shape = print_shape

        # Patch embedding: (B, L) → (B, num_patches, d_model)
        self.patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_prob,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final projection
        self.fc = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.ReLU(),
        )

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # EDA summary
        if eda_on_init:
            stats = eda_transformer_summary(self)
            print(stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, L) -> (B, latent_dim)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)

        # Patch embedding
        x = self.patch_embed(x).transpose(1, 2)  # (B, num_patches, d_model)
        x = x + self.pos_embed

        if self.print_shape:
            print(f"Patch embeddings: {x.shape}")

        # Transformer encoding
        x = self.encoder(x)
        if self.print_shape:
            print(f"Encoded sequence: {x.shape}")

        # Mean pooling across tokens
        x = x.mean(dim=1)  # (B, d_model)
        z = self.fc(x)  # (B, latent_dim)

        if self.print_shape:
            print(f"Latent vector: {z.shape}")

        return z
