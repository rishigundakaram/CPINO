from code.PINO.train_utils.losses import weighted_darcy_loss
import torch 
import torch.nn as nn
import torch.nn.functional as F

from CGDs import BCGD

from models.fourier2d import FNN2d
from models.fourier3d import FNN3d

from train_utils.losses import weighted_darcy_loss, darcy_loss

class CGD_PINO_2D(): 
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
        w_loss, uw_loss = weighted_darcy_loss(pred, a, w)
        self.w_loss = w_loss
        self.uw_loss = uw_loss
        return w_loss

    def step(self): 
        self.optimizer.step(loss=self.w_loss)
