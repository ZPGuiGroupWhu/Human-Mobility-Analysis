import torch


def to_var(var, device):
    if torch.is_tensor(var):
        var = var.to(device)
        return var
    if isinstance(var, int) or isinstance(var, float):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key], device)
        return var
    if isinstance(var, list):
        var = list(map(lambda x: to_var(x, device), var))
        return var