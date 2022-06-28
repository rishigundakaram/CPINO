import torch 
import torch.nn as nn
import torch.nn.functional as F

from CGDs import BCGD

from .FNO import FNN2d

class CPINO(): 
    def __init__(self, params) -> None:
        Weighter = FNN2d(params)
        Regressor = FNN2d(params)
        optimizer = BCGD(max_params=Weighter.params(), 
                        min_params=Regressor.params(), 
                        lr_min=params['lr_min'], 
                        lr_max=params['lr_max'],
                        momentum=params['momentum'])
        w_loss = None
        uw_loss = None
    def __call__(self, x):
        return self.Regressor(x)
    
    def loss(self, pred, a): 
        w = self.Weighter(a)
        w_loss = None
        uw_loss = None
        self.w_loss = w_loss
        self.uw_loss = uw_loss
        return w_loss

    def step(self): 
        self.optimizer.step(loss=self.w_loss)

    