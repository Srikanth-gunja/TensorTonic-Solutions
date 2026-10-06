import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 probability tensor with the same shape as logits.
    """
    # logits=logits-logits.max()
    # return torch.exp(logits)/torch.exp(logits).sum(dim=-1,keepdim=True)
    # logits = logits.to(torch.float32)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    exp = torch.exp(logits)
    return exp / exp.sum(dim=-1, keepdim=True)