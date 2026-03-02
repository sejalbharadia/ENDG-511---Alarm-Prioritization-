import torch
import torch.nn as nn
from tqdm import tqdm
try:
    from lightly.loss import NTXentLoss
except Exception:
    NTXentLoss = None


class SSLWrapper(nn.Module):
    """Simple SSL wrapper using backbone + projection head compatible with Lightly loss.

    This wrapper does not include augmentations; the user should provide an augmented
    dataloader that yields two views per sample: (x1, x2).
    """

    def __init__(self, backbone, proj_hidden=128, proj_out=64):
        super().__init__()
        self.backbone = backbone
        # projection head: expects backbone to provide final feature map as last element
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.backbone.out_features, proj_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(proj_hidden, proj_out),
        )

    def forward(self, x):
        # returns projection vector
        _, _, _, f4 = self.backbone(x)
        z = self.proj(f4)
        z = nn.functional.normalize(z, dim=1)
        return z


def ssl_train(wrapper: SSLWrapper, dataloader, optimizer, device='cpu', epochs=10, temperature=0.5):
    if NTXentLoss is None:
        raise RuntimeError("lightly is required for NTXentLoss; install lightly to run ssl_train")

    wrapper.to(device)
    criterion = NTXentLoss(temperature=temperature)

    for epoch in range(epochs):
        wrapper.train()
        pbar = tqdm(dataloader, desc=f"SSL E{epoch}")
        for batch in pbar:
            # assume batch is (x1, x2)
            x1, x2 = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            optimizer.zero_grad()
            z1 = wrapper(x1)
            z2 = wrapper(x2)
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': float(loss.item())})

    return wrapper
