import torch
import numpy as np
import math
from math import pi, gamma, sqrt
from torch.fft import ifft
from train_utils.loss import PINO_FDM
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def get_grid3d(S, T, time_scale=1.0, device='cpu'):
    gridx = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1 * time_scale, T), dtype=torch.float, device=device)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])
    return gridx, gridy, gridt

def get_forcing(S):
    x1 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(S, 1).repeat(1, S)
    x2 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(1, S).repeat(S, 1)
    return -4 * (torch.cos(4*(x2))).reshape(1,S,S,1)

class GaussianRF(object):
    def __init__(self, dim, size, length=1.0, alpha=2.0, tau=3.0, sigma=None, boundary="periodic", constant_eig=False, device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        const = (4*(math.pi**2))/(length**2)

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((const*(k**2) + tau**2)**(-alpha/2.0))

            if constant_eig:
                self.sqrt_eig[0] = size*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((const*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))

            if constant_eig:
                self.sqrt_eig[0,0] = (size**2)*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((const*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))

            if constant_eig:
                self.sqrt_eig[0,0,0] = (size**3)*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig*coeff

        u = torch.fft.irfftn(coeff, self.size, norm="backward")
        return u


def convert_ic(u0, N, S, T, time_scale=1.0):
    u0 = u0.reshape(N, S, S, 1, 1).repeat([1, 1, 1, T, 1])
    gridx, gridy, gridt = get_grid3d(S, T, time_scale=time_scale, device=u0.device)
    a_data = torch.cat((gridx.repeat([N, 1, 1, 1, 1]), gridy.repeat([N, 1, 1, 1, 1]),
                        gridt.repeat([N, 1, 1, 1, 1]), u0), dim=-1)
    return a_data

def online_loader(sampler, S, T, time_scale, batchsize=1):
    while True:
        u0 = sampler.sample(batchsize)
        a = convert_ic(u0, batchsize,
                       S, T,
                       time_scale=time_scale)
        yield a
        
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class Plotter: 
    def __init__(self, forcing, v, t_interval, save_dir='./CPINO/bin/', sample=0) -> None:
        self.cur_step = 0
        self.save_dir = save_dir
        self.sample = sample
        self.forcing = forcing
        self.v = v
        self.t_interval = t_interval
    
    def __animate_tensor(self, tensor, title): 
        fig, ax = plt.subplots()
        def animate(i):
            ax.imshow(tensor[0, ..., i])
            ax.set_axis_off()
        anim = FuncAnimation(fig, animate)
        writergif = PillowWriter(fps=30) 
        print(self.save_dir + title)
        anim.save(self.save_dir + title, writer=writergif)
        
    def step(self, output, u, a):
        print(u.size(), output["output"].size())
        data_error = (u - output["output"])[0, ...]
        print(data_error.size())
        u0  = a[:, :, :, 0, -1]
        ic_truth, ic_pred, f_truth, f_pred = PINO_FDM(output['output'], u0, self.forcing, self.v, self.t_interval)
        ic_weights, f_weights = output["ic_weights"], output["f_weights"] 
        print(ic_truth.size(), ic_pred.size(), f_truth.size(), f_pred.size())
        print(ic_weights.size(), f_weights.size())
        plt.matshow(ic_weights[0, ...].cpu())
        plt.colorbar()
        plt.savefig(self.save_dir + f'ic_weights_{self.cur_step}.png')
        plt.cla()
        plt.matshow(ic_truth[0, ...].cpu())
        plt.colorbar()
        plt.savefig(self.save_dir + f'ic_truth_{self.cur_step}.png')
        plt.cla()
        plt.matshow(ic_pred[0, ...].cpu())
        plt.colorbar()
        plt.savefig(self.save_dir + f'ic_pred_{self.cur_step}.png')
        self.__animate_tensor(data_error, f'error_{self.cur_step}.gif')
        self.cur_step += 1
        
        
        
        