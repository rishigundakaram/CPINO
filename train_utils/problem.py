from random import sample
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from .utils import get_grid3d, get_forcing
from math import floor

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

    def loss(self, input, target, prediction, bets=None): 
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
        if 'valid_data' in config.keys():
            valid_dataset = wave1Ddataset(
                config['valid_data']['path'], 
                sub_x=config['valid_data']['sub_x'],
                sub_t=config['valid_data']['sub_t']
                )
            self.valid_loader = DataLoader(
                valid_dataset, 
                batch_size=1, 
                shuffle=False,
            )
        
    def _FDM_Wave(self, u, D=1, c=1.0):
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


    def physics_truth(self, u, u0, c=1.0, **kwargs):
        batchsize = u.size(0)
        nt = u.size(1)
        nx = u.size(2)

        boundary_u = u[:, 0, :, 0]
        u = u.reshape(batchsize, nt, nx)
        # lploss = LpLoss(size_average=True)

        # index_t = torch.zeros(nx,).long()
        # index_x = torch.tensor(range(nx)).long()
        

        Du = self._FDM_Wave(u, c=c)[:, :, :]
        f = torch.zeros(Du.shape, device=u.device)
        return u0[:, 0, :, 0], boundary_u, f, Du

        # loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
        # loss_bc1 = F.mse_loss((u[:, :, 1] - u[:, :, -1]) /
        #                       (2/(nx)), (u[:, :, 0] - u[:, :, -2])/(2/(nx)))
    
class NS3D(problem): 
    def __init__(self, config) -> None:
        super().__init__(config)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        train_config = config['train_data']
        path_1 = train_config['path1']
        path_2 = None
        if 'path2' in train_config.keys(): 
            path_2 = train_config['path2']
        train_loader = NSDataset(datapath1=path_1, datapath2=path_2,
                          nx=train_config['nx'], nt=train_config['nt'],
                          sub=train_config['sub_x'], sub_t=train_config['sub_t'],
                          t_interval=train_config['time_interval'])
        
        train_num_samples = train_loader.data.size()[0]
        if 'valid_data' in config.keys(): 
            sample_proportion = config['valid_data']['sample_proportion']
            offset = floor(sample_proportion*train_num_samples)
            train_num_samples -= offset

            self.valid_loader = train_loader.make_loader(offset,
                                batch_size=config['train_params']['batchsize'],
                                start=train_num_samples,
                                train=train_config['shuffle'])

        self.train_loader = train_loader.make_loader(train_num_samples,
                                batch_size=config['train_params']['batchsize'],
                                start=0,
                                train=True)
        test_config = config['test_data']
        test_loader = NSDataset(datapath1=test_config['path'],
                          nx=test_config['nx'], nt=test_config['nt'],
                          sub=test_config['sub_x'], sub_t=test_config['sub_t'],
                          t_interval=test_config['time_interval'])
        self.test_loader = test_loader.make_loader(test_loader.data.size()[0],
                                batch_size=1,
                                start=0,
                                train=False)
        self.v = 1 / config['train_data']['Re']
        self.train_forcing = get_forcing(train_config['nx'] // train_config['sub_x']).to(device)
        self.test_forcing = get_forcing(test_config['nx'] // test_config['sub_x']).to(device)
        self.train_t_interval = train_config['time_interval']
        self.test_t_interval = test_config['time_interval']
    
    def _FDM(self, w, v=1/40, t_interval=1.0):
        batchsize = w.size(0)
        nx = w.size(1)
        ny = w.size(2)
        nt = w.size(3)
        device = w.device
        w = w.reshape(batchsize, nx, ny, nt)

        w_h = torch.fft.fft2(w, dim=[1, 2])
        # Wavenumbers in y-direction
        k_max = nx//2
        N = nx
        k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                        torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1, N).reshape(1,N,N,1)
        k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                        torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N, 1).reshape(1,N,N,1)
        # Negative Laplacian in Fourier space
        lap = (k_x ** 2 + k_y ** 2)
        lap[0, 0, 0, 0] = 1.0
        f_h = w_h / lap

        ux_h = 1j * k_y * f_h
        uy_h = -1j * k_x * f_h
        wx_h = 1j * k_x * w_h
        wy_h = 1j * k_y * w_h
        wlap_h = -lap * w_h

        ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
        uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
        wx = torch.fft.irfft2(wx_h[:, :, :k_max+1], dim=[1,2])
        wy = torch.fft.irfft2(wy_h[:, :, :k_max+1], dim=[1,2])
        wlap = torch.fft.irfft2(wlap_h[:, :, :k_max+1], dim=[1,2])

        dt = t_interval / (nt-1)
        wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)

        Du1 = wt + (ux*wx + uy*wy - v*wlap)[...,1:-1] #- forcing
        return Du1


    def physics_truth(self, u, u0, **kwargs):
        batchsize = u.size(0)
        nx = u.size(1)
        ny = u.size(2)
        nt = u.size(3)

        u = u.reshape(batchsize, nx, ny, nt)
        v = kwargs['v']
        t_interval = kwargs['t_interval']
        forcing = kwargs['forcing']
        u_in = u[:, :, :, 0]

        Du = self._FDM(u, v, t_interval)
        f = forcing.repeat(batchsize, 1, 1, nt-2)
        return u0[:, :, :, 0, 3], u_in, f, Du


class NSDataset(object):
    def __init__(self, datapath1,
                 nx, nt,
                 datapath2=None, sub=1, sub_t=1, t_interval=1.0):
        '''
        Load data from npy and reshape to (N, X, Y, T)
        Args:
            datapath1: path to data
            nx:
            nt:
            datapath2: path to second part of data, default None
            sub:
            sub_t:
            N:
            t_interval:
        '''
        self.S = nx // sub
        self.T = int(nt * t_interval) // sub_t + 1
        self.time_scale = t_interval
        data1 = np.load(datapath1)
        data1 = torch.tensor(data1, dtype=torch.float)[..., ::sub_t, ::sub, ::sub]
        if datapath2 is not None:
            data2 = np.load(datapath2)
            data2 = torch.tensor(data2, dtype=torch.float)[..., ::sub_t, ::sub, ::sub]

        if t_interval == 0.5:
            data1 = self.extract(data1)
            if datapath2 is not None:
                data2 = self.extract(data2)
        part1 = data1.permute(0, 2, 3, 1)
        if datapath2 is not None:
            part2 = data2.permute(0, 2, 3, 1)
            self.data = torch.cat((part1, part2), dim=0)
        else:
            self.data = part1

    def make_loader(self, n_sample, batch_size, start=0, train=True):
        if train:
            a_data = self.data[start:start + n_sample, :, :, 0].reshape(n_sample, self.S, self.S)
            u_data = self.data[start:start + n_sample].reshape(n_sample, self.S, self.S, self.T)
        else:
            a_data = self.data[-n_sample:, :, :, 0].reshape(n_sample, self.S, self.S)
            u_data = self.data[-n_sample:].reshape(n_sample, self.S, self.S, self.T)
        a_data = a_data.reshape(n_sample, self.S, self.S, 1, 1).repeat([1, 1, 1, self.T, 1])
        gridx, gridy, gridt = get_grid3d(self.S, self.T, time_scale=self.time_scale)
        a_data = torch.cat((gridx.repeat([n_sample, 1, 1, 1, 1]), gridy.repeat([n_sample, 1, 1, 1, 1]),
                            gridt.repeat([n_sample, 1, 1, 1, 1]), a_data), dim=-1)
        dataset = torch.utils.data.TensorDataset(a_data, u_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
        return loader

    def make_dataset(self, n_sample, start=0, train=True):
        if train:
            a_data = self.data[start:start + n_sample, :, :, 0].reshape(n_sample, self.S, self.S)
            u_data = self.data[start:start + n_sample].reshape(n_sample, self.S, self.S, self.T)
        else:
            a_data = self.data[-n_sample:, :, :, 0].reshape(n_sample, self.S, self.S)
            u_data = self.data[-n_sample:].reshape(n_sample, self.S, self.S, self.T)
        a_data = a_data.reshape(n_sample, self.S, self.S, 1, 1).repeat([1, 1, 1, self.T, 1])
        gridx, gridy, gridt = get_grid3d(self.S, self.T)
        a_data = torch.cat((
            gridx.repeat([n_sample, 1, 1, 1, 1]),
            gridy.repeat([n_sample, 1, 1, 1, 1]),
            gridt.repeat([n_sample, 1, 1, 1, 1]),
            a_data), dim=-1)
        dataset = torch.utils.data.TensorDataset(a_data, u_data)
        return dataset

    @staticmethod
    def extract(data):
        '''
        Extract data with time range 0-0.5, 0.25-0.75, 0.5-1.0, 0.75-1.25,...
        Args:
            data: tensor with size N x 129 x 128 x 128

        Returns:
            output: (4*N-1) x 65 x 128 x 128
        '''
        T = data.shape[1] // 2
        interval = data.shape[1] // 4
        N = data.shape[0]
        new_data = torch.zeros(4 * N - 1, T + 1, data.shape[2], data.shape[3])
        for i in range(N):
            for j in range(4):
                if i == N - 1 and j == 3:
                    # reach boundary
                    break
                if j != 3:
                    new_data[i * 4 + j] = data[i, interval * j:interval * j + T + 1]
                else:
                    new_data[i * 4 + j, 0: interval] = data[i, interval * j:interval * j + interval]
                    new_data[i * 4 + j, interval: T + 1] = data[i + 1, 0:interval + 1]
        return new_data

class Loss():
    def __init__(self, config, physics_truth, p=2, forcing=None, v=None, t_interval=None):
        self.model = config['info']['model']
        self.physics = physics_truth
        self.ic_weight = config['train_params']['ic_loss']
        self.f_weight = config['train_params']['f_loss']
        self.data_weight = config['train_params']['xy_loss']
        self.formulation = config['info']['formulation']
        self.loss = config['train_params']['loss']
        self.forcing = forcing
        self.v = v
        self.t_interval = t_interval
        self.p = p

    def L2(self, x, y, weights=None):
        num_examples = x.size()[0]
        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)
        if weights is None: 
            all_norms = h*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        else:
            size = weights.size()
            w = weights.expand(num_examples,*size)
            sq = (x.view(num_examples,-1) - y.view(num_examples,-1))**2
            all_norms = w.view(num_examples, -1)*sq
        return torch.mean(all_norms)
    
    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        return torch.mean(diff_norms/y_norms)


    def L2_physics_loss(self, u, u0): 
        ic_truth, ic_pred, f_truth, f_pred = self.physics(u, u0, forcing=self.forcing, v=self.v, t_interval=self.t_interval)
        ic_loss = self.L2(ic_truth, ic_pred)
        f_loss = self.L2(f_truth, f_pred) 
        return ic_loss, f_loss

    def w_physics_loss(self, u, u0, ic_weights, f_weights): 
        ic_truth, ic_pred, f_truth, f_pred = self.physics(u, u0, forcing=self.forcing, v=self.v, t_interval=self.t_interval)
        if self.model == "SAPINO" or self.model == "SAPINN": 
            ic_loss_w = self.L2(ic_truth, ic_pred, weights=ic_weights)
        else: 
            ic_loss_w = self.cLoss(ic_truth, ic_pred, ic_weights)
        
        if self.model == "SAPINO" or self.model == "SAPINN": 
            f_loss_w =self.L2(f_truth, f_pred, weights=f_weights)
        else: 
            f_loss_w = self.cLoss(f_truth, f_pred, f_weights)
        return ic_loss_w, f_loss_w

    def cLoss(self, x, y, weights):
        num_examples = x.size()[0]
        errs_w = torch.mean(weights.reshape(num_examples, -1) * (x.reshape(num_examples, -1)-y.reshape(num_examples, -1)))
        return errs_w
    
    def __call__(self, input, target, prediction):
        output = prediction["output"]
        loss = {}
        data_loss = self.L2(output, target)
        rel_data_loss = self.rel(output, target)
        ic_loss, f_loss = self.L2_physics_loss(output, input)
        loss["rel data loss"] = rel_data_loss.item()
        loss["L2 data loss"] = data_loss.item()
        loss["L2 ic loss"] = ic_loss.item()
        loss["L2 f loss"] = f_loss.item()
        loss["L2 loss"] = self.ic_weight * ic_loss.item() + self.f_weight * f_loss.item() + self.data_weight * data_loss.item()
        if self.model in ["CPINN", "CPINO", "SAPINN", "SAPINO", "CPINO-split"]: 
            ic_weights = prediction["ic_weights"]
            f_weights = prediction["f_weights"]
            if self.formulation == 'competitive': 
                data_weights = prediction['data_weights']
                data_loss = self.cLoss(output, target, weights=data_weights)
                loss["weighted data loss"] = data_loss.item()
            ic_loss_w, f_loss_w, = self.w_physics_loss(
                output, input, ic_weights, f_weights
                )
            if self.loss == 'rel': 
                loss["loss"] = self.ic_weight * ic_loss_w + self.f_weight * f_loss_w + self.data_weight * rel_data_loss
            else: 
                loss["loss"] = self.ic_weight * ic_loss_w + self.f_weight * f_loss_w + self.data_weight * data_loss
            loss["weighted f err"] =  f_loss_w.item()
            loss["weighted ic err"] = ic_loss_w.item()
        else:
            if self.loss == 'rel': 
                loss["loss"] = self.ic_weight * ic_loss + self.f_weight * f_loss + self.data_weight * rel_data_loss
            else: 
                loss["loss"] = self.ic_weight * ic_loss + self.f_weight * f_loss + self.data_weight * data_loss
        return loss