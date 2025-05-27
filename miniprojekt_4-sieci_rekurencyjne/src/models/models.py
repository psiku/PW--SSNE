import torch
import torch.nn as nn



# simple GRU model
class SimpleGRUClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=128,
        output_size=5,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        factor = 2 if bidirectional else 1
        self.norm = nn.LayerNorm(hidden_size * factor)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * factor, output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        out, _ = self.gru(x)  # [batch, seq_len, hidden_size * factor]

        pooled = out.mean(dim=1)  # [batch, hidden_size * factor]
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)  #  [batch, output_size]

        return logits

# model with embeddings
class GRUClassifier(nn.Module):
    def __init__(
        self,
        num_acords,
        embedding_dim=32,
        hidden_size=128,
        output_size=5,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    ):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=num_acords, embedding_dim=embedding_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        factor = 2 if bidirectional else 1
        self.norm = nn.LayerNorm(hidden_size * factor)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * factor, output_size)

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        out, _ = self.gru(x)   # (batch_size, seq_len, hidden_size * factor)

        pooled = out.mean(dim=1)  # (batch_size, hidden_size * factor)

        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)

        logits = self.fc(pooled)  # (batch_size, output_size)
        return logits
