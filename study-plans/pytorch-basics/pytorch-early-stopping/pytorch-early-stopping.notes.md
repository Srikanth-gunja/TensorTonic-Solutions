import torch

import torch.nn as nn

def train_with_early_stopping(model: nn.Module, train_loader: [torch.utils.data](http://torch.utils.data).DataLoader, val_loader: [torch.utils.data](http://torch.utils.data).DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, max_epochs: int, patience: int) -&gt; dict:

    """

    Returns train_losses and val_losses as float lists, and stopped_epoch as an int.

    """

    

    prev_loss=float("inf")

    var_pat=patience

    cur_params=None

    isEarlyStopped=False

    stooped_epoch=max_epochs

    train_losses=[]

    cur_loss=0.0

    val_losses=[]

    for epoch in range(max_epochs):

        model.train()

        running_loss=0.0

        if var_pat==0:

            stooped_epoch=epoch

            isEarlyStopped=True

            break   # got best one 

        for X,y in train_loader:

            y_pred=model(X)

            L=criterion(y_pred,y)

            [optimizer.zero](http://optimizer.zero)_grad()

            L.backward()

            optimizer.step() # update 

            running_loss+=L.item()

            epoch_loss=running_loss/len(train_loader)

        train_losses.append(epoch_loss)

        cur_loss=epoch_loss

      

        if cur_loss&lt;prev_loss:

            # upadte patience,update prev_loss

            prev_loss=cur_loss

            var_pat=patience

            cur_params=model.parameters

            # track parameters of that loss

        else:

            var_pat-=1

    

        if isEarlyStopped:

            model.parameters=cur_params

        model.eval()

        for x,y in val_loader:

            with [torch.no](http://torch.no)_grad():

                r_loss=0.0

                L=criterion(model(x),y)

                r_loss+=L.item()

        val_losses.append(r_loss/len(val_loader))

        

    

    return { "train_losses":train_losses[0:stooped_epoch],"val_losses":val_losses[0:stooped_epoch], "stopped_epoch":stooped_epoch}

    # if isEarlyStopped:

    #     model.parameters=cur_params

    # else:

    #     model.eval()