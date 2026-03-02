import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self,
        n_points,
        nb_conv_layers=3,
        filters_per_layer=[32, 64, 128],
        kernel_sizes=[20, 20, 20],
        dropout_rate_conv=0.25,
        nb_dense_layers=2,
        dense_units=[128, 1],
        dropout_rate_fc=[0.4, 0.3],
        pool_size=3,
    ):
        super().__init__()

        self.n_points = n_points
        self.nb_conv_layers = nb_conv_layers
        self.nb_dense_layers = nb_dense_layers
        self.conv_dropout = nn.Dropout(dropout_rate_conv)
        self.conv_layers = nn.ModuleList()
        self.fc_dropout = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dense_layers = nn.ModuleList()
        self.relu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

        in_channels = 1
        for i in range(nb_conv_layers):
            conv_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=filters_per_layer[i],
                kernel_size=kernel_sizes[i],
                stride=1,
                padding=0,
                dilation=1,
            )
            self.conv_layers.append(conv_layer)
            self.pool_layers.append(nn.MaxPool1d(pool_size))
            in_channels = filters_per_layer[i]

        self.flattened_size = self._get_flattened_size()

        for i in range(nb_dense_layers):
            if i == 0:
                dense_layer = nn.Linear(self.flattened_size, dense_units[i])
            else:
                dense_layer = nn.Linear(dense_units[i - 1], dense_units[i])
            self.dense_layers.append(dense_layer)
            self.fc_dropout.append(nn.Dropout(dropout_rate_fc[i]))

    def _get_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_points)
            for i in range(self.nb_conv_layers):
                x = self.conv_layers[i](x)
                x = self.pool_layers[i](x)
            return x.numel()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Convolutional layers
        for i in range(self.nb_conv_layers):
            x = self.conv_layers[i](x)
            x = self.pool_layers[i](x)
            x = self.relu(x)
            x = self.conv_dropout(x)

        x = x.view(x.size(0), -1)
        # Dense layers
        for i in range(self.nb_dense_layers - 1):
            x = self.dense_layers[i](x)
            x = self.relu(x)
            x = self.fc_dropout[i](x)

        x = self.dense_layers[-1](x)

        return x


class LSTM(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        num_layers=2,
        output_size=1,
        window_size=25,
        stride=1,
        dropout=0.2,
    ):
        super().__init__()

        self.window_size = window_size
        self.stride = stride

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(2)

        batch_size, total_len, dims = x.shape

        if total_len > self.window_size:
            x = x.unfold(1, self.window_size, self.stride)
            x = x.contiguous().view(-1, self.window_size, dims)
            is_unfolded = True
        else:
            is_unfolded = False

        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x = (x - mean) / (std + 1e-8)  # Numerical stability

        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])

        if is_unfolded:
            out = out.view(batch_size, -1)
            out = out.mean(dim=1, keepdim=True)

        return out
