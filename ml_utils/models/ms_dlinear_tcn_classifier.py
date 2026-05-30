# Ideas borrowed from existing project model modules:
# - Reuse DLinear-style seasonal/trend decomposition with replicate padding.
# - Reuse causal TCN padding/chomp structure with residual projections.

from collections.abc import Sequence

import torch
import torch.nn as nn


class MultiScaleDLinearTCNClassifier(nn.Module):
    """Multi-scale DLinear plus causal TCN classifier.

    Args:
        seq_len: Number of input time steps.
        input_size: Number of features per time step.
        num_classes: Number of output classes.
        moving_avg_kernels: Positive odd moving-average kernels for scales.
        tcn_channels: Output channels for each temporal convolution block.
        tcn_kernel_size: Conv1d kernel size for the TCN branch.
        dropout: Dropout probability used in the TCN branch.
    """

    def __init__(
        self,
        seq_len: int,
        input_size: int,
        num_classes: int = 2,
        moving_avg_kernels: Sequence[int] = (3, 5, 9, 15),
        tcn_channels: Sequence[int] = (16, 16),
        tcn_kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _validate_positive_int("seq_len", seq_len)
        _validate_positive_int("input_size", input_size)
        _validate_positive_int("num_classes", num_classes)
        kernels = _validate_moving_avg_kernels(moving_avg_kernels)
        channels = _validate_positive_int_sequence("tcn_channels", tcn_channels)
        _validate_positive_int("tcn_kernel_size", tcn_kernel_size)
        _validate_dropout(dropout)

        self.seq_len = seq_len
        self.input_size = input_size
        self.num_classes = num_classes
        self.moving_avg_kernels = tuple(kernels)
        self.tcn_channels = tuple(channels)
        self.tcn_kernel_size = tcn_kernel_size
        self.dropout = dropout

        self.dlinear_scales = nn.ModuleList(
            _DLinearScale(seq_len=seq_len, input_size=input_size, kernel_size=kernel)
            for kernel in kernels
        )
        self.dlinear_projection = nn.Linear(
            len(kernels) * seq_len * input_size,
            channels[-1],
        )
        self.tcn_branch = _TCNEncoder(
            input_size=input_size,
            channels=channels,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
        )
        self.fusion_classifier = nn.Linear(channels[-1] * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for input windows.

        Args:
            x: torch.Tensor of shape (batch, seq_len, input_size).

        Returns:
            torch.Tensor of shape (batch, num_classes).
        """
        self._validate_input_shape(x)

        scale_outputs = [scale(x) for scale in self.dlinear_scales]
        dlinear_features = torch.cat(
            [output.reshape(output.shape[0], -1) for output in scale_outputs],
            dim=1,
        )
        dlinear_repr = self.dlinear_projection(dlinear_features)
        tcn_repr = self.tcn_branch(x)
        fused = torch.cat([dlinear_repr, tcn_repr], dim=1)
        return self.fusion_classifier(fused)

    def _validate_input_shape(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(
                "Expected 3D input shape (batch, seq_len, input_size), "
                f"got {x.shape}"
            )
        if x.shape[1] != self.seq_len:
            raise ValueError(
                f"Expected seq_len={self.seq_len} in input shape, got {x.shape}"
            )
        if x.shape[2] != self.input_size:
            raise ValueError(
                "Expected input_size/feature dimension "
                f"{self.input_size} in input shape, got {x.shape}"
            )


class _DLinearScale(nn.Module):
    """DLinear scale preserving shape (batch, seq_len, input_size)."""

    def __init__(self, seq_len: int, input_size: int, kernel_size: int) -> None:
        super().__init__()
        self.decomposition = _SeriesDecomposition(kernel_size)
        self.seasonal_linear = nn.Linear(seq_len, seq_len)
        self.trend_linear = nn.Linear(seq_len, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scale features for an input tensor.

        Args:
            x: torch.Tensor of shape (batch, seq_len, input_size).

        Returns:
            torch.Tensor of shape (batch, seq_len, input_size).
        """
        seasonal, trend = self.decomposition(x)
        seasonal_output = self.seasonal_linear(seasonal.transpose(1, 2))
        trend_output = self.trend_linear(trend.transpose(1, 2))
        return (seasonal_output + trend_output).transpose(1, 2)


class _SeriesDecomposition(nn.Module):
    """Split tensors into seasonal and trend components."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_average = _MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return seasonal and trend tensors.

        Args:
            x: torch.Tensor of shape (batch, seq_len, input_size).

        Returns:
            Tuple of tensors with shape (batch, seq_len, input_size).
        """
        trend = self.moving_average(x)
        seasonal = x - trend
        return seasonal, trend


class _MovingAverage(nn.Module):
    """Moving average over tensors of shape (batch, seq_len, input_size)."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) // 2
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return moving averages for an input tensor.

        Args:
            x: torch.Tensor of shape (batch, seq_len, input_size).

        Returns:
            torch.Tensor of shape (batch, seq_len, input_size).
        """
        assert x.dim() == 3, (
            "Expected 3D input shape (batch, seq_len, input_size), "
            f"got {x.shape}"
        )
        if self.padding > 0:
            front = x[:, 0:1, :].repeat(1, self.padding, 1)
            end = x[:, -1:, :].repeat(1, self.padding, 1)
            x = torch.cat([front, x, end], dim=1)
        return self.avg_pool(x.transpose(1, 2)).transpose(1, 2)


class _TCNEncoder(nn.Module):
    """Causal TCN encoder returning the last-step representation."""

    def __init__(
        self,
        input_size: int,
        channels: Sequence[int],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
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
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final TCN representation.

        Args:
            x: torch.Tensor of shape (batch, seq_len, input_size).

        Returns:
            torch.Tensor of shape (batch, channels[-1]).
        """
        features = self.backbone(x.transpose(1, 2))
        return features[:, :, -1]


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
        """Return block output for a channel-first sequence tensor.

        Args:
            x: torch.Tensor of shape (batch, channels, seq_len).

        Returns:
            torch.Tensor of shape (batch, out_channels, seq_len).
        """
        return self.activation(self.net(x) + self.residual(x))


class _Chomp1d(nn.Module):
    """Remove right-side padding from tensor of shape (batch, channels, seq_len)."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return tensor with causal padding removed.

        Args:
            x: torch.Tensor of shape (batch, channels, padded_seq_len).

        Returns:
            torch.Tensor of shape (batch, channels, seq_len).
        """
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


def _validate_moving_avg_kernels(values: Sequence[int]) -> list[int]:
    kernels = _validate_positive_int_sequence("moving_avg_kernels", values)
    for index, kernel in enumerate(kernels):
        if kernel % 2 == 0:
            raise ValueError(
                f"moving_avg_kernels[{index}] must be odd, got {kernel}"
            )
    return kernels


def _validate_positive_int_sequence(name: str, values: Sequence[int]) -> list[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a non-empty sequence of positive ints")
    result = list(values)
    if not result:
        raise ValueError(f"{name} must be a non-empty sequence of positive ints")
    for index, value in enumerate(result):
        _validate_positive_int(f"{name}[{index}]", value)
    return result


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value}")


def _validate_dropout(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value < 1
    ):
        raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {value}")
