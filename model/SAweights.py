from distutils.command.config import config
import torch 
import torch.nn as nn
import torch.nn.functional as F

from CGDs import ACGD

from .PINN import simpleLinear
from .PINO import FNN2d, FNN3d
from .basics import SpectralConv2d, Model
from train_utils.adam import Adam, NAdam


class SAWeights(nn.Module):
    def __init__(self, dims, nx, nt, sub_x, sub_t, normalize=True):
        super(SAWeights, self).__init__()
        if dims == 4: 
            self.ic_params = nn.Parameter(torch.ones((nx//sub_x,nx//sub_x)))
            self.f_params =  nn.Parameter(torch.ones(nx//sub_x, nx//sub_x, (nt-1)//sub_t))
        else: 
            self.ic_params = nn.Parameter(torch.ones(nx//sub_x))
            self.f_params =  nn.Parameter(torch.ones((nt-1)//sub_t, nx//sub_x,))

    def forward(self):
        return self.ic_params, self.f_params

class SAPINN(Model):
    def __init__(self, params) -> None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cur_params = params['model_1']
        data_params = params['train_data']
        if 'modes3' in cur_params: 
            self.model = simpleLinear(indim=4, outdim=1, layers=cur_params['layers']).to(device)
            self.weight_model = SAWeights(
                dims=4, 
                nx=data_params['nx'],
                nt=int(data_params['nt']*data_params['time_interval']),
                sub_x=data_params['sub_x'],
                sub_t=data_params['sub_t']
                ).to(device)
        else: 
            self.model = simpleLinear(indim=3, outdim=1, layers=cur_params['layers']).to(device)
            self.weight_model = SAWeights(
                dims=3, 
                nx=data_params['nx'],
                nt=data_params['nt']*data_params['time_interval'],
                sub_x=data_params['sub_x'],
                sub_t=data_params['sub_t']
                ).to(device)
        self.optimizer = Adam(self.model.parameters(), betas=(0.9, 0.999),
                     lr=params['train_params']['lr_min'])
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                     milestones=params['train_params']['milestones'],
                                                     gamma=params['train_params']['scheduler_gamma'])

        self.weight_optimizer = NAdam(self.weight_model.parameters(), betas=(0.9, 0.999),
                     lr=params['train_params']['lr_max'])
        self.weight_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.weight_optimizer,
                                                     milestones=params['train_params']['milestones'],
                                                     gamma=params['train_params']['scheduler_gamma'])
    def train(self): 
        self.model.train()
        self.weight_model.train()
    
    def eval(self):
        self.model.eval()
        self.weight_model.eval()

    def step(self, loss):
        loss["loss"].backward()
        self.optimizer.step() 
        self.optimizer.zero_grad()
        self.weight_optimizer.step() 
        self.weight_optimizer.zero_grad()
    
    def schedule_step(self): 
        self.scheduler.step()
        self.weight_scheduler.step()
    
    def predict(self, x, dof=1): 
        out = self.model(x)
        ret = {}
        ret["ic_weights"], ret["f_weights"] = self.weight_model()
        ret["output"] = out
        return ret
    
    def __call__(self, x):
        return self.model(x)

class SAPINO(Model):
    def __init__(self, params) -> None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cur_params = params['model_1']
        data_params = params['train_data']
        if 'modes3' in cur_params: 
            self.model =  FNN3d(
                modes1=cur_params['modes1'], 
                modes2=cur_params['modes2'], 
                modes3=cur_params['modes3'], 
                fc_dim=cur_params['fc_dim'], 
                layers=cur_params['layers'],
                in_dim=4, 
            ).to(device)
            self.weight_model = SAWeights(
                dims=4, 
                nx=data_params['nx'],
                nt=int(data_params['nt']*data_params['time_interval']),
                sub_x=data_params['sub_x'],
                sub_t=data_params['sub_t']
                ).to(device)
        else: 
            self.model =  FNN2d(
                modes1=cur_params['modes1'], 
                modes2=cur_params['modes2'], 
                fc_dim=cur_params['fc_dim'], 
                layers=cur_params['layers'],
                activation=cur_params['activation']
            ).to(device)
            self.weight_model = SAWeights(
                dims=3, 
                nx=data_params['nx'],
                nt=data_params['nt'],
                sub_x=data_params['sub_x'],
                sub_t=data_params['sub_t']
                ).to(device)
        self.optimizer = Adam(self.model.parameters(), betas=(0.9, 0.999),
                     lr=params['train_params']['lr_min'])
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                     milestones=params['train_params']['milestones'],
                                                     gamma=params['train_params']['scheduler_gamma'])

        self.weight_optimizer = NAdam(self.weight_model.parameters(), betas=(0.9, 0.999),
                     lr=params['train_params']['lr_max'])
        self.weight_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.weight_optimizer,
                                                     milestones=params['train_params']['milestones'],
                                                     gamma=params['train_params']['scheduler_gamma'])
    def train(self): 
        self.model.train()
        self.weight_model.train()
    
    def eval(self):
        self.model.eval()
        self.weight_model.eval()

    def step(self, loss):
        loss["loss"].backward()
        self.optimizer.step() 
        self.optimizer.zero_grad()
        self.weight_optimizer.step() 
        self.weight_optimizer.zero_grad()
    
    def schedule_step(self): 
        self.scheduler.step()
        self.weight_scheduler.step()
    
    def predict(self, x, dof=1): 
        out = self.model(x)
        ret = {}
        ret["ic_weights"], ret["f_weights"] = self.weight_model()
        ret["output"] = out
        return ret
    
    def __call__(self, x):
        return self.model(x)
    