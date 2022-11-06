from random import sample
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
import numpy as np
from .utils import get_grid3d, get_forcing, GaussianRF, online_loader
from math import floor, pi
from .loss import PINO_FDM


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
        train_config = config['data']
        dataset = NS3DDataset(
            paths=config['data']['paths'], 
            raw_res=config['data']['raw_res'],
            data_res=config['data']['data_res'], 
            pde_res=config['data']['pde_res'], 
            n_samples=config['data']['n_samples'], 
            offset=config['data']['offset'], 
            t_duration=config['data']['t_duration']
            )
        
        idxs = torch.randperm(len(dataset))
        # setup train and test
        num_valid = config['data']['n_valid_samples']
        num_train = len(idxs) - num_valid
        print(f'Number of training samples: {num_train};\nNumber of validation samples: {num_valid}.')
        train_idx = idxs[:num_train]
        test_idx = idxs[num_train:]

        trainset = Subset(dataset, indices=train_idx)
        valset = Subset(dataset, indices=test_idx)
        
        batchsize = config['train']['batchsize']
        self.train_loader = DataLoader(trainset, batch_size=batchsize, num_workers=4, shuffle=True)

        self.valid_loader = DataLoader(valset, batch_size=batchsize, num_workers=4)
        
        if config['train_params']['tts_batchsize'] > 0: 
            tts_sampler = GaussianRF(2, config['tts_train_data']['nx'], 2 * pi, alpha=2.5, tau=7, device=device)
            self.tts_loader = online_loader(tts_sampler,
                             S=config['tts_train_data']['nx'],
                             T=config['tts_train_data']['nt'],
                             time_scale=config['tts_train_data']['time_interval'],
                             batchsize=config['train_params']['tts_batchsize'])
        self.v = 1 / config['train_data']['Re']
        self.forcing = get_forcing(train_config['nx'] // train_config['sub_x']).to(device)
        self.physics_truth = PINO_FDM
        self.train_t_interval = train_config['time_interval']
    

class NS3DDataset(Dataset):
    def __init__(self, paths, 
                 data_res, pde_res,
                 n_samples=None, 
                 offset=0,
                 t_duration=1.0, 
                 sub_x=1, 
                 sub_t=1,
                 train=True):
        super().__init__()
        self.data_res = data_res
        self.pde_res = pde_res
        self.t_duration = t_duration
        self.paths = paths
        self.offset = offset
        self.n_samples = n_samples
        self.load(train=train, sub_x=sub_x, sub_t=sub_t)
    
    def load(self, train=True, sub_x=1, sub_t=1):
        data_list = []
        for datapath in self.paths:
            batch = np.load(datapath, mmap_mode='r+')

            batch = torch.from_numpy(batch[:, ::sub_t, ::sub_x, ::sub_x]).to(torch.float32)
            if self.t_duration == 0.5:
                batch = self.extract(batch)
            data_list.append(batch.permute(0, 2, 3, 1))
        data = torch.cat(data_list, dim=0)
        if self.n_samples:
            if train:
                data = data[self.offset: self.offset + self.n_samples]
            else:
                data = data[self.offset + self.n_samples:]
        
        N = data.shape[0]
        S = data.shape[1]
        T = data.shape[-1]
        a_data = data[:, :, :, 0:1, None].repeat([1, 1, 1, T, 1])
        gridx, gridy, gridt = get_grid3d(S, T)
        a_data = torch.cat((
            gridx.repeat([N, 1, 1, 1, 1]),
            gridy.repeat([N, 1, 1, 1, 1]),
            gridt.repeat([N, 1, 1, 1, 1]),
            a_data), dim=-1)
        self.data = data        # N, S, S, T, 1
        self.a_data = a_data    # N, S, S, T, 4
        
        self.data_s_step = data.shape[1] // self.data_res[0]
        self.data_t_step = data.shape[3] // (self.data_res[2] - 1)

    def __getitem__(self, idx):
        return self.data[idx, ::self.data_s_step, ::self.data_s_step, ::self.data_t_step], self.a_data[idx]

    def __len__(self, ):
        return self.data.shape[0]

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

class KF(problem): 
    def __init__(self, config) -> None:
        super().__init__(config)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        train_config = config['data']
        batchsize = config['train_params']['batchsize']
        u_set = KFDataset(paths=config['data']['paths'], 
                          raw_res=config['data']['raw_res'],
                          data_res=config['data']['data_res'], 
                          pde_res=config['data']['data_res'], 
                          n_samples=config['data']['n_data_samples'], 
                          offset=config['data']['offset'], 
                          t_duration=config['data']['t_duration'])
        self.u_loader = DataLoader(u_set, batch_size=batchsize, num_workers=1, shuffle=True)

        a_set = KFaDataset(paths=config['data']['paths'], 
                           raw_res=config['data']['raw_res'], 
                           pde_res=config['data']['pde_res'], 
                           n_samples=config['data']['n_a_samples'],
                           offset=config['data']['a_offset'], 
                           t_duration=config['data']['t_duration'])
        self.a_loader = DataLoader(a_set, batch_size=batchsize, num_workers=1, shuffle=True)
        # val set
        valset = KFDataset(paths=config['data']['paths'], 
                           raw_res=config['data']['raw_res'],
                           data_res=config['test']['data_res'], 
                           pde_res=config['test']['data_res'], 
                           n_samples=config['data']['n_test_samples'], 
                           offset=config['data']['testoffset'], 
                           t_duration=config['data']['t_duration'])
        self.val_loader = DataLoader(valset, batch_size=1, num_workers=1)
        print(f'Train set: {len(u_set)}; Test set: {len(valset)}; IC set: {len(a_set)}')
        nx = config['data']['pde_res'][0]
        nt = config['data']['pde_res'][2]
        if config['train_params']['tts_batchsize'] > 0: 
            
            tts_sampler = GaussianRF(2, nx, 2 * pi, alpha=2.5, tau=7, device=device)
            self.tts_loader = online_loader(tts_sampler,
                             S=nx,
                             T=nt,
                             time_scale=config['data']['t_duration'],
                             batchsize=config['train_params']['tts_batchsize'])
        self.v = 1 / config['data']['Re']
        self.forcing = get_forcing(nx).to(device)
        print(self.forcing.size())
        self.physics_truth = PINO_FDM
        self.train_t_interval = train_config['t_duration']

class KFDataset(Dataset):
    def __init__(self, paths, 
                 data_res, pde_res, 
                 raw_res, 
                 n_samples=None, 
                 offset=0,
                 t_duration=1.0):
        super().__init__()
        self.data_res = data_res    # data resolution
        self.pde_res = pde_res      # pde loss resolution
        self.raw_res = raw_res      # raw data resolution
        self.t_duration = t_duration
        self.paths = paths
        self.offset = offset
        self.n_samples = n_samples
        if t_duration == 1.0:
            self.T = self.pde_res[2]
        else:
            self.T = int(self.pde_res[2] * t_duration) + 1    # number of points in time dimension

        self.load()

        self.data_s_step = pde_res[0] // data_res[0]
        self.data_t_step = (pde_res[2] - 1) // (data_res[2] - 1)

    def load(self):
        datapath = self.paths[0]
        raw_data = np.load(datapath, mmap_mode='r')
        # subsample ratio
        sub_x = self.raw_res[0] // self.data_res[0]
        sub_t = (self.raw_res[2] - 1) // (self.data_res[2] - 1)
        
        a_sub_x = self.raw_res[0] // self.pde_res[0]
        # load data
        data = raw_data[self.offset: self.offset + self.n_samples, ::sub_t, ::sub_x, ::sub_x]
        # divide data
        if self.t_duration != 0.:
            end_t = self.raw_res[2] - 1
            K = int(1/self.t_duration)
            step = end_t // K
            data = self.partition(data)
            a_data = raw_data[self.offset: self.offset + self.n_samples, 0:end_t:step, ::a_sub_x, ::a_sub_x]
            a_data = a_data.reshape(self.n_samples * K, 1, self.pde_res[0], self.pde_res[1])    # 2N x 1 x S x S
        else:
            a_data = raw_data[self.offset: self.offset + self.n_samples, 0:1, ::a_sub_x, ::a_sub_x]

        # convert into torch tensor
        data = torch.from_numpy(data).to(torch.float32)
        a_data = torch.from_numpy(a_data).to(torch.float32).permute(0, 2, 3, 1)
        self.data = data.permute(0, 2, 3, 1)

        S = self.pde_res[1]
        
        a_data = a_data[:, :, :, :, None]   # N x S x S x 1 x 1
        gridx, gridy, gridt = get_grid3d(S, self.T)
        self.grid = torch.cat((gridx[0], gridy[0], gridt[0]), dim=-1)   # S x S x T x 3
        self.a_data = a_data

    def partition(self, data):
        '''
        Args:
            data: tensor with size N x T x S x S
        Returns:
            output: int(1/t_duration) *N x (T//2 + 1) x 128 x 128
        '''
        N, T, S = data.shape[:3]
        K = int(1 / self.t_duration)
        new_data = np.zeros((K * N, T // K + 1, S, S))
        step = T // K
        for i in range(N):
            for j in range(K):
                new_data[i * K + j] = data[i, j * step: (j+1) * step + 1]
        return new_data


    def __getitem__(self, idx):
        a_data = torch.cat((
            self.grid, 
            self.a_data[idx].repeat(1, 1, self.T, 1)
        ), dim=-1)
        return self.data[idx], a_data

    def __len__(self, ):
        return self.data.shape[0]

class KFaDataset(Dataset):
    def __init__(self, paths, 
                 pde_res, 
                 raw_res, 
                 n_samples=None, 
                 offset=0,
                 t_duration=1.0):
        super().__init__()
        self.pde_res = pde_res      # pde loss resolution
        self.raw_res = raw_res      # raw data resolution
        self.t_duration = t_duration
        self.paths = paths
        self.offset = offset
        self.n_samples = n_samples
        if t_duration == 1.0:
            self.T = self.pde_res[2]
        else:
            self.T = int(self.pde_res[2] * t_duration) + 1    # number of points in time dimension

        self.load()

    def load(self):
        datapath = self.paths[0]
        raw_data = np.load(datapath, mmap_mode='r')
        # subsample ratio
        a_sub_x = self.raw_res[0] // self.pde_res[0]
        # load data
        if self.t_duration != 0.:
            end_t = self.raw_res[2] - 1
            K = int(1/self.t_duration)
            step = end_t // K
            a_data = raw_data[self.offset: self.offset + self.n_samples, 0:end_t:step, ::a_sub_x, ::a_sub_x]
            a_data = a_data.reshape(self.n_samples * K, 1, self.pde_res[0], self.pde_res[1])    # 2N x 1 x S x S
        else:
            a_data = raw_data[self.offset: self.offset + self.n_samples, 0:1, ::a_sub_x, ::a_sub_x]

        # convert into torch tensor
        a_data = torch.from_numpy(a_data).to(torch.float32).permute(0, 2, 3, 1)
        S = self.pde_res[1]
        a_data = a_data[:, :, :, :, None]   # N x S x S x 1 x 1
        gridx, gridy, gridt = get_grid3d(S, self.T)
        self.grid = torch.cat((gridx[0], gridy[0], gridt[0]), dim=-1)   # S x S x T x 3
        self.a_data = a_data

    def __getitem__(self, idx):
        a_data = torch.cat((
            self.grid, 
            self.a_data[idx].repeat(1, 1, self.T, 1)
        ), dim=-1)
        return a_data

    def __len__(self, ):
        return self.a_data.shape[0]