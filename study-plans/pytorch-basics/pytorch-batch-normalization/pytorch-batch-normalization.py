import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as X.
    """
    # Cal Mean ,Varinace 
    x_mean=X.mean(dim=0)
    x_var=X.var(dim=0,unbiased=False)

    x_new=(X-x_mean)/torch.sqrt(x_var+eps) # Normalize
    return (gamma*x_new)+beta
