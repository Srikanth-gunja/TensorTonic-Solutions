import torch
import torch.nn as nn

def manual_train_step(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.Module, lr: float) -> float:
    """
    Returns the pre-update batch loss as a Python float.
    """
    # w_init=model.weight
    # y_pred=model(X)
    # L=criterion(y_pred,y)
    # L.backward()
    # grad=model.weight.grad
    # model.weight=torch.nn.Parameter(w_init-lr*grad)

    # model.zero_grad()
    # y_pred=model(X)
    # L=criterion(y_pred,y)
    # return L.item()

    L=criterion(model(X),y)
    L.backward()
    loss=L.item()
    with torch.no_grad():
        for p in model.parameters():
            p-=lr*p.grad
    model.zero_grad()
    return loss
    
    
    