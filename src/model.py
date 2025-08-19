import torch
import torch.nn as nn

class SmallConvPolicy(nn.Module):
    def __init__(self, in_ch: int = 25, width: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head_from  = nn.Linear(width, 64)
        self.head_to    = nn.Linear(width, 64)
        self.head_promo = nn.Linear(width, 5)

    def forward(self, x):
        feat = self.net(x).flatten(1)  # [B, width]
        return {
            "from":  self.head_from(feat),
            "to":    self.head_to(feat),
            "promo": self.head_promo(feat),
        }
