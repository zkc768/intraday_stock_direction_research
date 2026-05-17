import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """LSTM classifier for window tensors.

    Args:
        input_size: Number of features per time step.
        hidden_size: LSTM hidden size.
        num_layers: Number of stacked LSTM layers.
        num_classes: Number of output classes.
        dropout: LSTM dropout probability, used only when num_layers > 1.
        bidirectional: Whether to use a bidirectional LSTM.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        _validate_positive("input_size", input_size)
        _validate_positive("hidden_size", hidden_size)
        _validate_positive("num_layers", num_layers)
        _validate_positive("num_classes", num_classes)
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}")

        self.input_size = input_size
        representation_size = hidden_size * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.layer_norm = nn.LayerNorm(representation_size)
        self.classifier = nn.Linear(representation_size, num_classes)

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

        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        normalized = self.layer_norm(last_step)
        return self.classifier(normalized)


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
