# Ideas borrowed from reference_excerpts/pytorch_tcn_core.py:
# - Preserve temporal length with padded Conv1d plus right-side chomp.
# - Grow dilation as 2 ** layer_index across stacked blocks.
# - Use a residual projection when block input and output channels differ.

from collections.abc import Sequence

import torch
import torch.nn as nn


class TCNClassifier(nn.Module):
    """Temporal convolution classifier for window tensors.

    Args:
        input_size: Number of features per time step.
        num_channels: Output channels for each temporal block.
        kernel_size: Conv1d kernel size.
        dropout: Dropout probability used inside temporal blocks.
        num_classes: Number of output classes.
        causal: Whether to use causal convolutions. Only causal=True is supported.
    """

    def __init__(
        self,
        input_size: int,
        num_channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
        num_classes: int = 2,
        causal: bool = True,
    ) -> None:
        super().__init__()
        _validate_positive_int("input_size", input_size)
        _validate_positive_int("kernel_size", kernel_size)
        _validate_positive_int("num_classes", num_classes)
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}")
        if causal is not True:
            raise ValueError("TCNClassifier currently supports causal=True only")

        channels = _validate_num_channels(num_channels)
        layers: list[nn.Module] = []
        in_channels = input_size
        for layer_index, out_channels in enumerate(channels):
            dilation = 2**layer_index
            layers.append(
                _TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.input_size = input_size
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for input windows.

        Args:
            x: torch.Tensor of shape (batch, seq_len, input_size).

        Returns:
            torch.Tensor of shape (batch, num_classes).
        """
        assert x.dim() == 3, (
            "Expected 3D input of shape "
            f"(batch, seq_len, input_size), got {x.shape}"
        )
        assert x.size(-1) == self.input_size, (
            f"Expected input_size={self.input_size}, got {x.size(-1)}"
        )
        if x.size(1) <= 0:
            raise ValueError(f"seq_len must be > 0, got {x.size(1)}")

        features = self.backbone(x.transpose(1, 2))
        last_step = features[:, :, -1]
        return self.classifier(last_step)


class _TemporalBlock(nn.Module):
    """Causal temporal block preserving sequence length."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            _Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            _Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if in_channels == out_channels:
            self.residual: nn.Module = nn.Identity()
        else:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return block output for tensor of shape (batch, channels, seq_len)."""
        return self.activation(self.net(x) + self.residual(x))


class _Chomp1d(nn.Module):
    """Remove right-side padding from tensor of shape (batch, channels, seq_len)."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return tensor with causal padding removed."""
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


def _validate_num_channels(num_channels: Sequence[int]) -> list[int]:
    if isinstance(num_channels, (str, bytes)) or not isinstance(num_channels, Sequence):
        raise ValueError("num_channels must be a non-empty sequence of positive ints")
    channels = list(num_channels)
    if not channels:
        raise ValueError("num_channels must be a non-empty sequence of positive ints")
    for index, value in enumerate(channels):
        _validate_positive_int(f"num_channels[{index}]", value)
    return channels


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value}")
