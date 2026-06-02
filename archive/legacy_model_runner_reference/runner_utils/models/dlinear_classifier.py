import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    """Moving average over tensors of shape (batch, seq_len, channels)."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        _validate_positive_odd_int("moving_avg_kernel", kernel_size)

        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return moving averages with shape (batch, seq_len, channels)."""
        assert x.dim() == 3, (
            "Expected 3D input shape (batch, seq_len, channels), "
            f"got {x.shape}"
        )
        if self.padding > 0:
            front = x[:, 0:1, :].repeat(1, self.padding, 1)
            end = x[:, -1:, :].repeat(1, self.padding, 1)
            x = torch.cat([front, x, end], dim=1)
        return self.avg_pool(x.transpose(1, 2)).transpose(1, 2)


class SeriesDecomposition(nn.Module):
    """Split tensors into seasonal and trend components."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tensors of shape (batch, seq_len, channels)."""
        trend = self.moving_average(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearClassifier(nn.Module):
    """TSLib-style DLinear adapted for classification.

    The read-only DLinear and Autoformer reference excerpts inform the
    decomposition and classification shape, but are not imported.

    Args:
        seq_len: Number of input time steps.
        input_size: Number of features per time step.
        num_classes: Number of output classes.
        moving_avg_kernel: Positive odd moving-average kernel size.
        individual: Whether each feature channel uses separate temporal linears.
    """

    def __init__(
        self,
        seq_len: int,
        input_size: int,
        num_classes: int = 2,
        moving_avg_kernel: int = 5,
        individual: bool = False,
    ) -> None:
        super().__init__()
        _validate_positive_int("seq_len", seq_len)
        _validate_positive_int("input_size", input_size)
        _validate_positive_int("num_classes", num_classes)
        _validate_positive_odd_int("moving_avg_kernel", moving_avg_kernel)
        if not isinstance(individual, bool):
            raise ValueError(f"individual must be bool, got {individual}")

        self.seq_len = seq_len
        self.input_size = input_size
        self.num_classes = num_classes
        self.moving_avg_kernel = moving_avg_kernel
        self.individual = individual
        self.decomposition = SeriesDecomposition(moving_avg_kernel)

        if individual:
            self.seasonal_linears = nn.ModuleList(
                nn.Linear(seq_len, seq_len) for _ in range(input_size)
            )
            self.trend_linears = nn.ModuleList(
                nn.Linear(seq_len, seq_len) for _ in range(input_size)
            )
        else:
            self.seasonal_linear = nn.Linear(seq_len, seq_len)
            self.trend_linear = nn.Linear(seq_len, seq_len)

        self.classifier = nn.Linear(seq_len * input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for input windows.

        Args:
            x: torch.Tensor of shape (batch, seq_len, input_size).

        Returns:
            torch.Tensor of shape (batch, num_classes).
        """
        self._validate_input_shape(x)

        seasonal, trend = self.decomposition(x)
        seasonal = seasonal.transpose(1, 2)
        trend = trend.transpose(1, 2)
        if self.individual:
            seasonal_output = seasonal.new_zeros(
                seasonal.size(0),
                self.input_size,
                self.seq_len,
            )
            trend_output = trend.new_zeros(
                trend.size(0),
                self.input_size,
                self.seq_len,
            )
            for channel in range(self.input_size):
                seasonal_output[:, channel, :] = self.seasonal_linears[channel](
                    seasonal[:, channel, :]
                )
                trend_output[:, channel, :] = self.trend_linears[channel](
                    trend[:, channel, :]
                )
        else:
            seasonal_output = self.seasonal_linear(seasonal)
            trend_output = self.trend_linear(trend)

        representation = (seasonal_output + trend_output).transpose(1, 2)
        flattened = representation.reshape(representation.shape[0], -1)
        return self.classifier(flattened)

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


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value}")


def _validate_positive_odd_int(name: str, value: int) -> None:
    _validate_positive_int(name, value)
    if value % 2 == 0:
        raise ValueError(f"{name} kernel must be odd, got {value}")
