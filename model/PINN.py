from distutils.command.config import config
import torch 
import torch.nn as nn
import torch.nn.functional as F

from CGDs import ACGD

from .PINO import FNN2d, FNN3d
from .basics import SpectralConv2d, Model
from train_utils.adam import Adam, NAdam


class simpleLinear(torch.nn.Module):
    def __init__(self, indim=3, outdim=1, layers=[64, 64, 64]):
        super(simpleLinear, self).__init__()
        mats = [indim] + layers + [outdim]
        self.linear = []
        for i, o in zip(mats, mats[1:]): 
            self.linear.append(torch.nn.Linear(i, o))
        self.linear = nn.ModuleList(self.linear)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        for layer in self.linear[:-1]: 
            x = layer(x)
            x = self.activation(x)
        x = self.linear[-1](x)
        return x

class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        if isinstance(nonlinearity, str):
            if nonlinearity == 'tanh':
                nonlinearity = nn.Tanh
            elif nonlinearity == 'relu':
                nonlinearity = nn.ReLU
            else:
                raise ValueError(f'{nonlinearity} is not supported')
        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

class PINN(Model):
    def __init__(self, params) -> None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cur_params = params['model_1']
        if 'modes3' in cur_params: 
            self.model = DenseNet(layers=cur_params['layers'], nonlinearity='relu').to(device)
        else: 
            layers = [3] + cur_params['layers'] + [1]
            self.model = DenseNet(layers, 'relu').to(device)
            # self.model = simpleLinear(indim=3, outdim=1, layers=cur_params['layers']).to(device)
            
        self.optimizer = Adam(self.model.parameters(), betas=(0.9, 0.999),
                     lr=params['train_params']['base_lr'])
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                     milestones=params['train_params']['milestones'],
                                                     gamma=params['train_params']['scheduler_gamma'])

    def step(self, loss):
        loss["loss"].backward()
        self.optimizer.step() 
        self.optimizer.zero_grad()
    
    def schedule_step(self): 
        self.scheduler.step()

