import torch

def compute_gradient(values: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 gradient tensor with the same shape as values.
    """
    values.requires_grad_(True)
    forward = torch.sum(values**3+2*values)
    forward.backward()
    return values.grad
