from numpy import dtype
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

class problem: 
    def __init__(self, config, d=2, p=2, size_average=True, reduction=True) -> None:
        self.ic_weight = config["train_params"]["ic_loss"]
        self.data_weight = config["train_params"]["xy_loss"]
        self.f_weight = config["train_params"]["f_loss"]
        
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.size_average = size_average
        self.reduction = reduction

    def loss(self,input, target, prediction, bets=None): 
        data_err =  torch.norm(prediction - target)
        loss = {
            "L2 data error": data_err, 
            "loss": self.data_weight * data_err
        }
        return loss

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms


class wave1Ddataset: 
    def __init__(self, path, sub_x=None, sub_t=None) -> None:
        device = torch.device('cpu')
        data = torch.load(path)
        ic = data['a'].to(device)
        sol = data['u'].to(device)
        Nsamples, Nt, Nx = sol.size()
        print(sol.size())
        if Nx % sub_x != 0 or Nt % sub_t != 0: 
            raise ValueError('invalid subsampling. Subsampling must be a whole number')
        Nt = int(Nt / sub_t)
        Nx = int(Nx / sub_x)

        ic = ic[:, ::sub_x]
        sol = sol[:, ::sub_t, ::sub_x]
        

        ic = ic[:, None, :]
        zeros = torch.zeros((Nsamples, Nt-1, Nx)).to(device)
        ic = torch.cat((ic, zeros), dim=1)
        x = torch.arange(start=0.0, end=1.0, step=1/Nx)
        t = torch.arange(start=0.0, end=1.0, step=1/Nt)
        grid_x, grid_t = torch.meshgrid((x, t))
        grid_x = grid_x[None, ...].repeat((Nsamples, 1, 1))
        grid_t = grid_t[None, ...].repeat((Nsamples, 1, 1))
        grid_x = torch.permute(grid_x, (0, 2, 1))
        grid_t = torch.permute(grid_t, (0, 2, 1))

        
        self.initial_conditions = torch.stack((ic, grid_x, grid_t), dim=-1).to(torch.float)
        self.solution = sol[..., None].to(torch.float)
        self.initial_conditions = self.initial_conditions.to(device)
        self.solution = self.solution.to(device)
    

    def __len__(self): 
        return self.solution.size()[0]

    def __getitem__(self, idx): 
        return (self.initial_conditions[idx, ...], self.solution[idx, ...])

class wave1D(problem): 
    def __init__(self, config) -> None:
        super().__init__(config)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        train_dataset = wave1Ddataset(
            config['train_data']['path'], 
            sub_x=config['train_data']['sub_x'],
            sub_t=config['train_data']['sub_t']
            )
        test_dataset = wave1Ddataset(
            config['test_data']['path'], 
            sub_x=config['test_data']['sub_x'],
            sub_t=config['test_data']['sub_t']
            )

        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config['train_params']['batchsize'], 
            shuffle=False,
            )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False,
        )
    
    def loss(self, input, target, prediction, bets=None):
        data_loss = self.abs(prediction, target)
        f_loss, ic_loss = self.PINO_loss_wave(prediction, input[:, 0, :, 0])
        loss = {
            "loss": self.ic_weight * ic_loss + self.f_weight * f_loss + self.data_weight * data_loss, 
            "L2 data err": data_loss,
            "L2 f err": f_loss, 
            "L2 ic err": ic_loss
        }
        return loss
    
    def FDM_Wave(self, u, D=1, c=1.0):
        batchsize = u.size(0)
        nt = u.size(1)
        nx = u.size(2)

        u = u.reshape(batchsize, nt, nx)
        dt = D / (nt-1)
        dx = D / (nx)

        u_h = torch.fft.fft(u, dim=2)
        # Wavenumbers in y-direction
        k_max = nx//2
        k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
                        torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(1,1,nx)
        ux_h = 2j *np.pi*k_x*u_h
        uxx_h = 2j *np.pi*k_x*ux_h
        ux = torch.fft.irfft(ux_h[:, :, :k_max+1], dim=2, n=nx)
        uxx = torch.fft.irfft(uxx_h[:, :, :k_max+1], dim=2, n=nx)
        ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
        utt = (u[:, 2:, :] - 2.0*u[:, 1:-1, :] + u[:, :-2, :]) / (dt**2)
        Du = utt - c**2 * uxx[:,1:-1,:]
        return Du


    def PINO_loss_wave(self, u, u0, c=1.0):
        batchsize = u.size(0)
        nt = u.size(1)
        nx = u.size(2)

        u = u.reshape(batchsize, nt, nx)
        # lploss = LpLoss(size_average=True)

        index_t = torch.zeros(nx,).long()
        index_x = torch.tensor(range(nx)).long()
        boundary_u = u[:, index_t, index_x]
        loss_u = F.mse_loss(boundary_u, u0)

        Du = self.FDM_Wave(u, c=c)[:, :, :]
        f = torch.zeros(Du.shape, device=u.device)
        loss_f = F.mse_loss(Du, f)

        # loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
        # loss_bc1 = F.mse_loss((u[:, :, 1] - u[:, :, -1]) /
        #                       (2/(nx)), (u[:, :, 0] - u[:, :, -2])/(2/(nx)))
        return loss_u, loss_f


    