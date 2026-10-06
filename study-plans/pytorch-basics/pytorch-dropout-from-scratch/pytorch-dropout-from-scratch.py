import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self,p: float = 0.5):
        super().__init__()

        # random value from uniform dis
        # if val <p =0 else keep element scale by 1/1-p
        # self.masked=torch.bernoulli()
        self.p=p
        # self.evalMode=evalMode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor with the same shape as x.
        """

        # if self.evalMode==True:
        #     print("hey")
        if not self.training or self.p==0:
            return x
        if self.p==1:
            return torch.zeros_like(x)
        return (x*(torch.rand_like(x)>=self.p))/(1-self.p)