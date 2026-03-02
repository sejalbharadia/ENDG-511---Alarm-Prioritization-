import torch
import os


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_example_input(batch_size=2, channels=4, time_steps=3000, device='cpu'):
    return torch.randn(batch_size, channels, time_steps, device=device)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
