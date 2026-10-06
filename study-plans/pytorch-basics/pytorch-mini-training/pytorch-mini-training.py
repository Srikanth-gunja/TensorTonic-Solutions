import torch
import torch.nn as nn

def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """
    Returns the mean batch loss as a Python float.
    """ 
    # dataloader contains [batch_size,features,target labels]
    model.train()                              # training mode
    losses = []

    for features, targets in dataloader:
        optimizer.zero_grad()                  # 1. wipe old gradients
        y_pred = model(features)               # 2. forward pass
        loss = criterion(y_pred, targets)      # 3. (prediction, target) order
        loss.backward()                        # 4. compute gradients
        optimizer.step()                       # 5. update weights

        losses.append(loss.item())             # .item() -> plain Python float

    return sum(losses) / len(losses)   
