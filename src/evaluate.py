import torch
import torch.nn.functional as F


def evaluate(model, dataloader, device='cpu'):
    model.to(device)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            logits = out['logits']
            preds = logits.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
    return correct / total if total > 0 else 0.0


def early_exit_inference(model, x, thresholds=(0.9, 0.9), device='cpu'):
    """Run a single-batch early-exit inference using confidence thresholds for exits.

    thresholds: tuple for (exit2_threshold, exit3_threshold)
    Returns dict with 'logits' and 'exit_level' (0=final,2,3)
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(x.to(device))
        # compute softmax confidences
        e2 = F.softmax(out['exit2'], dim=1)
        e3 = F.softmax(out['exit3'], dim=1)
        # max confidences
        c2, _ = e2.max(dim=1)
        c3, _ = e3.max(dim=1)

        # decide per-sample whether to exit early (vectorized)
        # if any sample wants to exit earlier, return that exit for that sample
        # For simplicity we return a single-level for full batch: prefer earliest where all samples exceed threshold
        if (c2 >= thresholds[0]).all():
            return {'logits': out['exit2'], 'exit_level': 2}
        if (c3 >= thresholds[1]).all():
            return {'logits': out['exit3'], 'exit_level': 3}
        return {'logits': out['logits'], 'exit_level': 4}
