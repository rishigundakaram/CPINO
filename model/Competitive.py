from itertools import chain
import torch 
import torch.nn as nn
import torch.nn.functional as F

from CGDs import ACGD

from .PINO import FNN2d, FNN3d
from .basics import SpectralConv2d, Model
from train_utils.adam import Adam, NAdam
from .PINN import simpleLinear



class CPINO(Model): 
    def __init__(self, params) -> None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_1_params = params['model_1']
        model_2_params = params['model_2']
        self.dim = 4 if 'modes3' in model_1_params else 3
        self.formulation = params['info']['formulation']
            
        if self.formulation == 'lagrangian':
            discriminator_out_dim = 2 
        else: 
            discriminator_out_dim = 3
        if self.dim == 4: 
            self.model = FNN3d(
                modes1=model_1_params['modes1'], 
                modes2=model_1_params['modes2'], 
                modes3=model_1_params['modes3'], 
                fc_dim=model_1_params['fc_dim'], 
                layers=model_1_params['layers'],
                in_dim=self.dim, 
                out_dim=1
            ).to(device)
            self.Discriminator = FNN3d(
                modes1=model_2_params['modes1'], 
                modes2=model_2_params['modes2'], 
                modes3=model_2_params['modes3'], 
                fc_dim=model_2_params['fc_dim'], 
                layers=model_2_params['layers'],
                in_dim=self.dim, 
                out_dim=discriminator_out_dim,
            ).to(device)
        else:
            self.model = FNN2d(
                modes1=model_1_params['modes1'], 
                modes2=model_1_params['modes2'], 
                fc_dim=model_1_params['fc_dim'], 
                layers=model_1_params['layers'],
                activation=model_1_params['activation'],
                in_dim=3
            ).to(device)
            self.Discriminator = FNN2d(
                modes1=model_2_params['modes1'], 
                modes2=model_2_params['modes2'],
                fc_dim=model_2_params['fc_dim'], 
                layers=model_2_params['layers'],
                activation=model_2_params['activation'],
                in_dim=3,
                out_dim=discriminator_out_dim,
            ).to(device)

        train_params = params['train_params']
        self.optimizer = ACGD(max_params=self.Discriminator.parameters(), 
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

    def predict(self, x): 
        out = self.model(x)
        out_w = self.Discriminator(x)
        ret = {}
        ret["output"] = out
        if self.dim == 3: 
            ret["ic_weights"] = out_w[..., 0,:,0]
            ret["f_weights"] = out_w[..., 1:-1, :, 1]
            if self.formulation == 'competitive':
                ret["data_weights"] = out_w[..., :, 2]
        else: 
            ret["ic_weights"] = out_w[..., 0,0]
            ret["f_weights"] = out_w[..., 1:-1, 1]
            if self.formulation == 'competitive':
                ret["data_weights"] = out_w[..., :, 2]
        return ret
    
    def save(self, path):
        super().save(path)
        path = path[:-3] + '-opt' + path[-3:]
        torch.save(self.optimizer.state_dict(), path)

class CPINN(Model): 
    def __init__(self, params) -> None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.dim = 3 if 'modes3' in params['model_1'] else 2
        if self.dim == 3: 
            self.model = simpleLinear(indim=4, outdim=1, layers=params['model_1']['layers']).to(device)
            self.Discriminator = simpleLinear(indim=4,outdim=3, layers=params['model_2']['layers']).to(device)
        else:
            self.model = simpleLinear(outdim=1, layers=params['model_1']['layers']).to(device)
            self.Discriminator = simpleLinear(outdim=3, layers=params['model_2']['layers']).to(device)

        train_params = params['train_params']
        self.optimizer = ACGD(max_params=self.Discriminator.parameters(), 
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
        if self.dim == 2: 
            ret["data_weights"] = out_w[..., 0]
            ret["ic_weights"] = out_w[..., 0,:,1]
            ret["f_weights"] = out_w[..., 1:-dof, :, 2]
        else: 
            ret["data_weights"] = out_w[..., 0]
            ret["ic_weights"] = out_w[..., 0,1]
            ret["f_weights"] = out_w[..., 1:-1, 2]
        return ret
    
class CPINO_SPLIT(Model): 
    def __init__(self, params) -> None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_1_params = params['model_1']
        model_2_params = params['model_2']
        model_3_params = params['model_3']
        model_4_params = params['model_4']
        self.dim = 4 if 'modes3' in model_1_params else 3
        self.formulation = params['info']['formulation']
        if self.dim == 4: 
            self.model = FNN3d(
                modes1=model_1_params['modes1'], 
                modes2=model_1_params['modes2'], 
                modes3=model_1_params['modes3'], 
                fc_dim=model_1_params['fc_dim'], 
                layers=model_1_params['layers'],
                in_dim=self.dim, 
                out_dim=1
            ).to(device)
            self.f_discriminator = FNN3d(
                modes1=model_2_params['modes1'], 
                modes2=model_2_params['modes2'], 
                modes3=model_2_params['modes3'], 
                fc_dim=model_2_params['fc_dim'], 
                layers=model_2_params['layers'],
                in_dim=4, 
                out_dim=1,
            ).to(device)
            self.ic_discriminator = FNN2d(
                modes1=model_3_params['modes1'], 
                modes2=model_3_params['modes2'], 
                fc_dim=model_3_params['fc_dim'], 
                layers=model_3_params['layers'],
                activation=model_3_params['activation'],
                in_dim=3,
            ).to(device)
            max_params = chain(self.f_discriminator.parameters(), self.ic_discriminator.parameters())
            if self.formulation == 'competitive': 
                self.data_discriminator = FNN3d(
                    modes1=model_4_params['modes1'], 
                    modes2=model_4_params['modes2'], 
                    modes3=model_4_params['modes3'], 
                    fc_dim=model_4_params['fc_dim'], 
                    layers=model_4_params['layers'],
                    in_dim=4, 
                    out_dim=1,
                ).to(device)
                max_params = chain(max_params, self.data_discriminator.parameters())
        else:
            raise ValueError('Can\'t use the split method on a 2D problem')

        train_params = params['train_params']
        self.optimizer = ACGD(max_params=max_params, 
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
        self.ic_discriminator.train()
        self.f_discriminator.train()
        if self.formulation == 'competitive': 
            self.data_discriminator.train()
    
    def eval(self):
        self.model.eval()
        self.ic_discriminator.eval()
        self.f_discriminator.eval()
        if self.formulation == 'competitive': 
            self.data_discriminator.eval()

    def predict(self, x, dof=1): 
        out = self.model(x)
        ret = {}
        ret["output"] = out
        ret["ic_weights"] = self.ic_discriminator(x[..., 0, [0,1,3]])
        ret["f_weights"] = self.f_discriminator(x)[...,dof:-dof, 0]
        if self.formulation == 'competitive': 
            ret["data_weights"] = self.data_discriminator(x)
        return ret

class CPINO_SIMGD(Model): 
    def __init__(self, params) -> None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_1_params = params['model_1']
        model_2_params = params['model_2']
        self.dim = 4 if 'modes3' in model_1_params else 3
        self.formulation = params['info']['formulation']
            
        if self.formulation == 'lagrangian':
            discriminator_out_dim = 2 
        else: 
            discriminator_out_dim = 3
        if self.dim == 4: 
            self.model = FNN3d(
                modes1=model_1_params['modes1'], 
                modes2=model_1_params['modes2'], 
                modes3=model_1_params['modes3'], 
                fc_dim=model_1_params['fc_dim'], 
                layers=model_1_params['layers'],
                in_dim=self.dim, 
                out_dim=1
            ).to(device)
            self.Discriminator = FNN3d(
                modes1=model_2_params['modes1'], 
                modes2=model_2_params['modes2'], 
                modes3=model_2_params['modes3'], 
                fc_dim=model_2_params['fc_dim'], 
                layers=model_2_params['layers'],
                in_dim=self.dim, 
                out_dim=discriminator_out_dim,
            ).to(device)
        else:
            self.model = FNN2d(
                modes1=model_1_params['modes1'], 
                modes2=model_1_params['modes2'], 
                fc_dim=model_1_params['fc_dim'], 
                layers=model_1_params['layers'],
                activation=model_1_params['activation'],
                in_dim=3
            ).to(device)
            self.Discriminator = FNN2d(
                modes1=model_2_params['modes1'], 
                modes2=model_2_params['modes2'],
                fc_dim=model_2_params['fc_dim'], 
                layers=model_2_params['layers'],
                activation=model_2_params['activation'],
                in_dim=3,
                out_dim=discriminator_out_dim,
            ).to(device)

        self.min_optimizer = Adam(self.model.parameters(), betas=(0.9, 0.999),
                     lr=params['train_params']['lr_min'])
        self.max_optimizer = NAdam(self.Discriminator.parameters(), betas=(0.9, 0.999),
                     lr=params['train_params']['lr_max'])
            
    def __call__(self, x):
        return self.model(x)

    def step(self, loss): 
        loss["loss"].backward()
        self.min_optimizer.step() 
        self.max_optimizer.step()
        self.min_optimizer.zero_grad()
        self.max_optimizer.zero_grad()

    def train(self): 
        self.model.train()
        self.Discriminator.train()
    
    def eval(self):
        self.model.eval()
        self.Discriminator.eval()

    def predict(self, x): 
        out = self.model(x)
        out_w = self.Discriminator(x)
        ret = {}
        ret["output"] = out
        if self.dim == 3: 
            ret["ic_weights"] = out_w[..., 0,:,0]
            ret["f_weights"] = out_w[..., 1:-1, :, 1]
            if self.formulation == 'competitive':
                ret["data_weights"] = out_w[..., :, 2]
        else: 
            ret["ic_weights"] = out_w[..., 0,0]
            ret["f_weights"] = out_w[..., 1:-1, 1]
            if self.formulation == 'competitive':
                ret["data_weights"] = out_w[..., :, 2]
        return ret
    
    def save(self, path):
        super().save(path)
        torch.save(self.optimizer.state_dict())

class CPINO_ALTGD(Model): 
    def __init__(self, params) -> None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_1_params = params['model_1']
        model_2_params = params['model_2']
        self.dim = 4 if 'modes3' in model_1_params else 3
        self.formulation = params['info']['formulation']
            
        if self.formulation == 'lagrangian':
            discriminator_out_dim = 2 
        else: 
            discriminator_out_dim = 3
        if self.dim == 4: 
            self.model = FNN3d(
                modes1=model_1_params['modes1'], 
                modes2=model_1_params['modes2'], 
                modes3=model_1_params['modes3'], 
                fc_dim=model_1_params['fc_dim'], 
                layers=model_1_params['layers'],
                in_dim=self.dim, 
                out_dim=1
            ).to(device)
            self.Discriminator = FNN3d(
                modes1=model_2_params['modes1'], 
                modes2=model_2_params['modes2'], 
                modes3=model_2_params['modes3'], 
                fc_dim=model_2_params['fc_dim'], 
                layers=model_2_params['layers'],
                in_dim=self.dim, 
                out_dim=discriminator_out_dim,
            ).to(device)
        else:
            self.model = FNN2d(
                modes1=model_1_params['modes1'], 
                modes2=model_1_params['modes2'], 
                fc_dim=model_1_params['fc_dim'], 
                layers=model_1_params['layers'],
                activation=model_1_params['activation'],
                in_dim=3
            ).to(device)
            self.Discriminator = FNN2d(
                modes1=model_2_params['modes1'], 
                modes2=model_2_params['modes2'],
                fc_dim=model_2_params['fc_dim'], 
                layers=model_2_params['layers'],
                activation=model_2_params['activation'],
                in_dim=3,
                out_dim=discriminator_out_dim,
            ).to(device)

        self.min_optimizer = Adam(self.model.parameters(), betas=(0.9, 0.999),
                     lr=params['train_params']['lr_min'])
        self.max_optimizer = NAdam(self.Discriminator.parameters(), betas=(0.9, 0.999),
                     lr=params['train_params']['lr_max'])
        self.opt_state = 1
            
    def __call__(self, x):
        return self.model(x)

    def step(self, loss): 
        loss["loss"].backward()
        if self.opt_state: 
            self.min_optimizer.step() 
        else: 
            self.max_optimizer.step()
        self.opt_state = not self.opt_state
        self.min_optimizer.zero_grad()
        self.max_optimizer.zero_grad()

    def train(self): 
        self.model.train()
        self.Discriminator.train()
    
    def eval(self):
        self.model.eval()
        self.Discriminator.eval()

    def predict(self, x): 
        out = self.model(x)
        out_w = self.Discriminator(x)
        ret = {}
        ret["output"] = out
        if self.dim == 3: 
            ret["ic_weights"] = out_w[..., 0,:,0]
            ret["f_weights"] = out_w[..., 1:-1, :, 1]
            if self.formulation == 'competitive':
                ret["data_weights"] = out_w[..., :, 2]
        else: 
            ret["ic_weights"] = out_w[..., 0,0]
            ret["f_weights"] = out_w[..., 1:-1, 1]
            if self.formulation == 'competitive':
                ret["data_weights"] = out_w[..., :, 2]
        return ret
    
    def save(self, path):
        super().save(path)
        path = path[:-3] + '-opt' + path[-3:]
        torch.save(self.optimizer.state_dict(), path)
    