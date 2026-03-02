import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class TinyBackbone(nn.Module):
    """Tiny 1D CNN backbone that returns intermediate features for early exits.

    Design constraints: pruning- and quantization-friendly (no inplace ops, simple modules).
    Keeps parameter count small (<500k) by using narrow channel widths.
    """

    def __init__(self, in_channels=4, channels=(16, 32, 64, 128)):
        super().__init__()
        assert len(channels) >= 3, "Need at least 3 blocks for early-exit features"
        self.layer1 = ConvBlock1D(in_channels, channels[0])
        self.layer2 = ConvBlock1D(channels[0], channels[1])
        self.layer3 = ConvBlock1D(channels[1], channels[2])
        self.layer4 = ConvBlock1D(channels[2], channels[3])

        # final embedding dim (after global pooling)
        self.out_features = channels[3]

    def forward(self, x):
        # x: (B, C, T)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        # return intermediate features for early exits
        # features are tensors (B, C, T')
        return f1, f2, f3, f4


class ClassifierHead(nn.Module):
    def __init__(self, in_channels, num_classes=2, reduction='avg'):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x: (B, C, T)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class EarlyExitHead(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, max(in_channels // 2, 16)),
            nn.ReLU(inplace=False),
            nn.Linear(max(in_channels // 2, 16), num_classes),
        )

    def forward(self, x):
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class AlarmClassifier(nn.Module):
    """Wrapper combining backbone + classifier head + early-exit heads.

    forward(x, early_exit=False, exit_thresholds=None) -> dict with keys:
      - logits: final logits
      - exits: list of early exit logits (None if not used)
      - features: intermediate features
    """

    def __init__(self, in_channels=4, num_classes=2, channels=(16, 32, 64, 128)):
        super().__init__()
        self.backbone = TinyBackbone(in_channels=in_channels, channels=channels)
        self.head = ClassifierHead(self.backbone.out_features, num_classes=num_classes)

        # early exit heads after layer2 and layer3
        self.exit2 = EarlyExitHead(channels[1], num_classes=num_classes)
        self.exit3 = EarlyExitHead(channels[2], num_classes=num_classes)

        # keep a small projection head for optional SSL usage
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.backbone.out_features, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 64),
        )

    def forward(self, x, return_features=False):
        f1, f2, f3, f4 = self.backbone(x)
        out = self.head(f4)
        e2 = self.exit2(f2)
        e3 = self.exit3(f3)

        result = {
            'logits': out,
            'exit2': e2,
            'exit3': e3,
            'features': (f1, f2, f3, f4),
        }

        if return_features:
            # also return global pooled final embedding
            result['embedding'] = self.projection(f4)

        return result


if __name__ == '__main__':
    # quick smoke test
    model = AlarmClassifier(in_channels=4, num_classes=2)
    x = torch.randn(2, 4, 3000)
    out = model(x)
    for k, v in out.items():
        if isinstance(v, tuple):
            print(k, [t.shape for t in v])
        else:
            print(k, None if v is None else v.shape)
