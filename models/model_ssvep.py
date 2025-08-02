import torch
import torch.nn as nn
import torch



class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=11, d_model=32, num_heads=4, num_layers=3, num_classes=4, dropout=0.1):
        super(TransformerClassifier, self).__init__()

        self.input_bn = nn.BatchNorm1d(input_dim)

        # Linear projection of input features to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding: learnable or sinusoidal
        self.pos_encoding = nn.Parameter(torch.randn(1, 1750, d_model))  # Assuming max length 1750

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_bn = nn.BatchNorm1d(d_model)
        self.fc = nn.Linear(d_model, 4)

    def forward(self, x):
        # x: [B, T, F] → BatchNorm
        x = self.input_bn(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T, F]
    
        x = self.input_proj(x)  # [B, T, d_model]
        x = x + self.pos_encoding[:, :x.size(1), :]  # Add positional encoding
    
        out = self.transformer(x)  # [B, T, d_model]
        pooled = out[:, -1, :]  # Use last token's representation
    
        pooled = self.output_bn(pooled)  # BN before final layer
        logits = self.fc(pooled)  # [B, 1]
        return logits
