import torch
import torch.nn as nn
import torch




class GRUClassifier(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, num_layers=3, num_classes=1, bidirectional=True):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # BatchNorm on input features (T is time, apply on F across B)
        self.input_bn = nn.BatchNorm1d(input_dim)

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.output_bn = nn.BatchNorm1d(hidden_dim * (2 if bidirectional else 1))
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        # x: [B, T, F] → permute to [B, F, T] for BN1d over F
        x = self.input_bn(x.permute(0, 2, 1)).permute(0, 2, 1)

        out, h_n = self.gru(x)

        if self.bidirectional:
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, 2H]
        else:
            h_n = h_n[-1]  # [B, H]

        h_n = self.output_bn(h_n)  # BN after GRU output
        logits = self.fc(h_n)
        return logits
