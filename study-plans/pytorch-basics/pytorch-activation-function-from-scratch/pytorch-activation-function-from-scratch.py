import torch

def activate(x: torch.Tensor, method: str = "relu") -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as x.
    """
    if method == "sigmoid":
        return 1 / (1 + torch.exp(-x))

    elif method == "tanh":
        return torch.tanh(x)

    elif method == "leaky_relu":
        return torch.where(x <= 0, 0.01 * x, x)

    else:  # relu
        return torch.where(x > 0, x, 0)
