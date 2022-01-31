import torch
import torch.nn.functional as F
from data_load import *

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target, smooth=1):
        batch_size = input.size()[0]
        loss = 0

        for idx in range(batch_size):
            coeff = self._coefficient(input[idx], target[idx])
            loss += coeff
        
        loss /= batch_size
        if batch_size == 1: return loss
        return 1 - loss

    def _coefficient(self, input, target, smooth=1):
        numer = 2*(torch.mul(input, target).sum()) + smooth
        denom = input.sum() + target.sum() + smooth
        coeff = numer/denom
        return coeff

class BCEDiceLoss(nn.Module):
    def __init__(self, device):
        super(BCEDiceLoss, self).__init__()
        self.dice_loss = DiceLoss().to(device)

    def forward(self, input, target):
        return F.binary_cross_entropy(input, target) + self.dice_loss(input, target)