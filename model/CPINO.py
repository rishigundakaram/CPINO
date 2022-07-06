import torch 
import torch.nn as nn
import torch.nn.functional as F

from CGDs import BCGD

from .FNO import FNN2d
from .basics import SpectralConv2d, Model
from train_utils.adam import Adam, NAdam

class CPINO(Model): 
    def __init__(self, params) -> None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model_1_params = params['model_1']
        self.model = FNN2d(
            modes1=model_1_params['modes1'], 
            modes2=model_1_params['modes2'], 
            fc_dim=model_1_params['fc_dim'], 
            layers=model_1_params['layers'],
            activation=model_1_params['activation'], 
        ).to(device)
        
        model_2_params = params['model_2']
        out_dim = len(model_2_params['competitive_output'])
        self.Discriminator = FNN2d(
            modes1=model_2_params['modes1'], 
            modes2=model_2_params['modes2'], 
            fc_dim=model_2_params['fc_dim'], 
            layers=model_2_params['layers'],
            activation=model_2_params['activation'],
            in_dim=3,
            out_dim=out_dim,
        ).to(device)

        train_params = params['train_params']
        self.optimizer = BCGD(max_params=self.Discriminator.parameters(), 
                        min_params=self.model.parameters(), 
                        lr_min=train_params['lr_min'], 
                        lr_max=train_params['lr_max'], 
                        tol=train_params["cg_tolerance"])
            
    def __call__(self, x):
        return self.model(x)

    def step(self, loss): 
        self.optimizer.step(loss=loss["loss"])
        self.optimizer.zero_grad()

    def train(self): 
        self.model.train()
        self.Discriminator.train()
    
    def eval(self):
        self.model.eval()
        self.Discriminator.eval()

    def predict(self, x, dof=1): 
        out = self.model(x)
        out_w = self.Discriminator(x)
        ret = {}
        ret["output"] = out
        ret["data_weights"] = out_w[..., 0]
        ret["ic_weights"] = out_w[..., 0,:,1]
        ret["f_weights"] = out_w[..., 1:-dof, :, 2]
        return ret

class CPINO(Model): 
    def __init__(self, params) -> None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model_1_params = params['model_1']
        self.model = FNN2d(
            modes1=model_1_params['modes1'], 
            modes2=model_1_params['modes2'], 
            fc_dim=model_1_params['fc_dim'], 
            layers=model_1_params['layers'],
            activation=model_1_params['activation'], 
        ).to(device)
        
        model_2_params = params['model_2']
        out_dim = len(model_2_params['competitive_output'])
        self.Discriminator = FNN2d(
            modes1=model_2_params['modes1'], 
            modes2=model_2_params['modes2'], 
            fc_dim=model_2_params['fc_dim'], 
            layers=model_2_params['layers'],
            activation=model_2_params['activation'],
            in_dim=3,
            out_dim=out_dim,
        ).to(device)

        self.optimizer_max = NAdam(self.Discriminator.parameters(), betas=(0.9, 0.999),
                     lr=params['train_params']['lr_max'])
        self.scheduler_max = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_max,
                                                     milestones=params['train_params']['milestones'],
                                                     gamma=params['train_params']['scheduler_gamma'])

        self.optimizer_min = Adam(self.model.parameters(), betas=(0.9, 0.999),
                     lr=params['train_params']['lr_min'])
        self.scheduler_min = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_min,
                                                     milestones=params['train_params']['milestones'],
                                                     gamma=params['train_params']['scheduler_gamma'])
            
    def __call__(self, x):
        return self.model(x)

    def step(self, loss): 
        loss["loss"].backward()
        self.optimizer_max.step() 
        self.optimizer_max.zero_grad()
        self.optimizer_min.step() 
        self.optimizer_min.zero_grad()

    def train(self): 
        self.model.train()
        self.Discriminator.train()
    
    def eval(self):
        self.model.eval()
        self.Discriminator.eval()

    def predict(self, x, dof=1): 
        out = self.model(x)
        out_w = self.Discriminator(x)
        ret = {}
        ret["output"] = out
        ret["data_weights"] = out_w[..., 0]
        ret["ic_weights"] = out_w[..., 0,:,1]
        ret["f_weights"] = out_w[..., 1:-dof, :, 2]
        return ret


    
    