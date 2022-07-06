from mimetypes import init
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
        data = torch.load(path, map_location=device)
        ic = data['a'].to(device)
        sol = data['u'].to(device)
        print("data is loaded in memory")
        Nsamples, Nt, Nx = sol.size()
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


    def physics_truth(self, u, u0, c=1.0):
        batchsize = u.size(0)
        nt = u.size(1)
        nx = u.size(2)

        boundary_u = u[:, 0, :, 0]
        u = u.reshape(batchsize, nt, nx)
        # lploss = LpLoss(size_average=True)

        # index_t = torch.zeros(nx,).long()
        # index_x = torch.tensor(range(nx)).long()
        

        Du = self.FDM_Wave(u, c=c)[:, :, :]
        f = torch.zeros(Du.shape, device=u.device)
        return u0[:, 0, :, 0], boundary_u, f, Du

        # loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
        # loss_bc1 = F.mse_loss((u[:, :, 1] - u[:, :, -1]) /
        #                       (2/(nx)), (u[:, :, 0] - u[:, :, -2])/(2/(nx)))

class Loss():
    def __init__(self, config, physics_truth, p=2):
        self.competitive = config['info']['competitive']
        self.physics = physics_truth
        self.ic_weight = config['train_params']['ic_loss']
        self.f_weight = config['train_params']['f_loss']
        self.data_weight = config['train_params']['xy_loss']
        self.p = p

    def L2(self, x, y):
        num_examples = x.size()[0]
        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = h*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        return torch.mean(all_norms)
    
    def L2_physics_loss(self, u, u0): 
        ic_truth, ic_pred, f_truth, f_pred = self.physics(u, u0)
        ic_loss = self.L2(ic_truth, ic_pred)
        f_loss = self.L2(f_truth, f_pred) 
        return ic_loss, f_loss

    def w_physics_loss(self, u, u0, ic_weights=None, f_weights=None): 
        ic_truth, ic_pred, f_truth, f_pred = self.physics(u, u0)
        if ic_weights is None: 
            ic_loss_w = self.L2(ic_truth, ic_pred)
        else: 
            ic_loss_w = self.cLoss(ic_truth, ic_pred, ic_weights)
        if f_weights is None: 
            f_loss_w =self.L2(f_truth, f_pred)
        else: 
            f_loss_w = self.cLoss(f_truth, f_pred, f_weights)
        return ic_loss_w, f_loss_w,

    def cLoss(self, x, y, weights=None):
        num_examples = x.size()[0]
        errs_w = torch.mean(weights.view(num_examples, -1) * (x.view(num_examples, -1)-y.view(num_examples, -1)))
        return errs_w
    
    def __call__(self, input, target, prediction):
        output = prediction["output"]
        loss = {}
        if self.competitive: 
            data_weights = prediction["data_weights"]
            ic_weights = prediction["ic_weights"]
            f_weights = prediction["f_weights"]

            data_loss_w = self.cLoss(output, target, weights=data_weights)
            ic_loss_w, f_loss_w, = self.w_physics_loss(
                output, input, ic_weights, f_weights
                )
            loss = {
                "loss": self.ic_weight * ic_loss_w + self.f_weight * f_loss_w + self.data_weight * data_loss_w, 
                "weighted data loss": data_loss_w.item(),
                "weighted f err": f_loss_w.item(), 
                "weighted ic err": ic_loss_w.item(),
            }
        L2_data_loss = self.L2(target, output)
        ic_loss, f_loss = self.L2_physics_loss(output, input)
        loss["L2 data loss"] = L2_data_loss.item()
        loss["L2 ic loss"] = ic_loss.item()
        loss["L2 f loss"] = f_loss.item()
        if not self.competitive: 
            loss["loss"] = self.ic_weight * ic_loss + self.f_weight * f_loss + self.data_weight * L2_data_loss
        return loss
            