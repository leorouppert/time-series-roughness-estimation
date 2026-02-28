import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self,
        N_points,
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

        self.N_points = N_points
        self.nb_conv_layers = nb_conv_layers
        self.filters_per_layer = filters_per_layer
        self.kernel_sizes = kernel_sizes
        self.dropout_rate_conv = dropout_rate_conv
        self.nb_dense_layers = nb_dense_layers
        self.dense_units = dense_units
        self.dropout_rate_fc = dropout_rate_fc
        self.pool_size = pool_size
        self.conv_layers = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dense_layers = nn.ModuleList()
        self.relu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

        in_channels = 1
        for i in range(self.nb_conv_layers):
            conv_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.filters_per_layer[i],
                kernel_size=self.kernel_sizes[i],
                stride=1,
                padding=0,
                dilation=1,
            )
            self.conv_layers.append(conv_layer)
            self.pool_layers.append(nn.MaxPool1d(self.pool_size))
            self.dropout.append(nn.Dropout(self.dropout_rate_conv))
            in_channels = self.filters_per_layer[i]

        self.flattened_size = self._get_flattened_size()

        for i in range(self.nb_dense_layers):
            if i == 0:
                dense_layer = nn.Linear(self.flattened_size, self.dense_units[i])
            else:
                dense_layer = nn.Linear(self.dense_units[i - 1], self.dense_units[i])
            self.dense_layers.append(dense_layer)
            self.dropout.append(nn.Dropout(self.dropout_rate_fc[i]))

    def _get_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.N_points)
            for i in range(self.nb_conv_layers):
                x = self.conv_layers[i](x)
                x = self.pool_layers[i](x)
            return x.numel()

    def forward(self, x):
        # Convolutional layers
        for i in range(self.nb_conv_layers):
            x = self.conv_layers[i](x)
            x = self.pool_layers[i](x)
            x = self.relu(x)
            x = self.dropout[i](x)

        x = x.view(x.size(0), -1)
        # Dense layers
        for i in range(self.nb_dense_layers - 1):
            x = self.dense_layers[i](x)
            x = self.relu(x)
            x = self.dropout[self.nb_conv_layers + i](x)

        x = self.dense_layers[-1](x)

        return self.sigmoid(x)
