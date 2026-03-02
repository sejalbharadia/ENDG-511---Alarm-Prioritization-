import torch
import torch_pruning as tp


def prune_model_l1(model, example_inputs, amount=0.3, device='cpu'):
    """Perform structured channel pruning using torch_pruning L1FilterPruner.

    Args:
        model: nn.Module
        example_inputs: a tuple with a single tensor example like (x,) or tensor
        amount: fraction of channels to prune per layer (0-1)
    Returns the pruned model (in-place).
    """
    model.to(device)
    model.eval()

    # build dependency graph
    if isinstance(example_inputs, tuple):
        example = example_inputs
    else:
        example = (example_inputs,)

    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example)

    # gather conv layers to prune
    pruning_plan = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv1d):
            pruning_plan.append(m)

    # apply L1 filter pruning on each conv layer proportionally
    for layer in pruning_plan:
        # number of filters to remove
        C_out = layer.out_channels
        n_prune = int(C_out * amount)
        if n_prune <= 0:
            continue
        strategy = tp.strategy.L1Strategy()
        pruner = tp.pruner.L1FilterPruner(model, example_inputs=example, ch_sparsity=amount, op_types=[torch.nn.Conv1d])
        pruner.step()

    return model
