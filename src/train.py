import torch
import torch.nn as nn
from tqdm import tqdm


def train_supervised(model, dataloader, optimizer, device='cpu', epochs=10, scheduler=None):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Train E{epoch}")
        total_loss = 0.0
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            logits = out['logits']
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})

    return model
