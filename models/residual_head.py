import torch
import torch.nn as nn

class QuantileResidualHead(nn.Module):
    def __init__(self, d_in, horizon, hidden1=128, hidden2=64, dropout=0.15):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 3 * horizon),  # p10, median, p90
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, 3, self.horizon)

def pinball_loss(y_true, y_pred_q, quantiles):
    y = y_true.unsqueeze(1).expand_as(y_pred_q)
    e = y - y_pred_q
    q = quantiles.view(1, -1, 1)
    return torch.maximum(q * e, (q - 1) * e).mean()

# sortiraj kvantile
def enforce_monotonic(y_q):
    return torch.sort(y_q, dim=1)[0]
