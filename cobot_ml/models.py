import torch
from torch import nn


class BiLSTM(nn.Module):
    def __init__(
            self,
            features_count: int = 25,
            hidden_size: int = 80,
            n_layers: int = 2,
            forecast_length: int = 10,
            dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.forecast_length = forecast_length
        self.n_layers = n_layers
        self.rnn = nn.LSTM(
            input_size=features_count,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Linear(hidden_size * 2, forecast_length)

    def __str__(self):
        return f"model=BiLSTM,layers={self.n_layers},forecast={self.forecast_length}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.linear(out)
        return out


class BiGRU(nn.Module):
    def __init__(
            self,
            features_count: int = 25,
            hidden_size: int = 80,
            n_layers: int = 2,
            forecast_length: int = 1,
            dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.forecast_length = forecast_length
        self.n_layers = n_layers
        self.rnn = nn.GRU(
            input_size=features_count,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Linear(hidden_size * 2, forecast_length)

    def __str__(self):
        return f"model=BiGRU,layers={self.n_layers},forecast={self.forecast_length}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.linear(out)
        return out


class LSTM(nn.Module):
    """
    Network created by Hundman in:
    https://arxiv.org/pdf/1802.04431.pdf

    It fits to the given data and predicts output_size steps ahead.
    Input shape is (batch_size, sequence_len, input_size)
    Output shape is (batch_size, output_size)

    :param features_count: Number of input features
    :param hidden_size: Number of features in hidden layers
    :param n_layers: Number of layers to stack
    :param forecast_length: Numbers of steps ahead to predict
    :param dropout_rate: Dropout rate after each but last LSTM cell
    """

    def __init__(
            self,
            features_count: int,
            hidden_size: int = 80,
            n_layers: int = 2,
            forecast_length: int = 1,
            dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.forecast_length = forecast_length
        self.n_layers = n_layers
        self.rnn = nn.LSTM(
            input_size=features_count,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, forecast_length)

    def __str__(self):
        return f"model=LSTM,layers={self.n_layers},forecast={self.forecast_length}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.linear(out)
        return out


class GRU(nn.Module):

    def __init__(
            self,
            features_count: int = 25,
            hidden_size: int = 80,
            n_layers: int = 2,
            forecast_length: int = 1,
            dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.forecast_length = forecast_length
        self.n_layers = n_layers
        self.rnn = nn.GRU(
            input_size=features_count,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, forecast_length)

    def __str__(self):
        return f"model=GRU,layers={self.n_layers},forecast={self.forecast_length}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.linear(out)
        return out
