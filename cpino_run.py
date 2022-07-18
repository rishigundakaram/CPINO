import yaml
import torch
from argparse import ArgumentParser
from tqdm import tqdm

from code.CPINO.model.PINO import FNN2d
from torch.utils.data import DataLoader
from train_utils.adam import Adam, NAdam
import numpy as np
from CGDs import BCGD, ACGD
import torch.nn.functional as F
import torch

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('seaborn-pastel')




class simpleLinear(torch.nn.Module):
    def __init__(self, outdim=1):
        super(simpleLinear, self).__init__()

        self.linear1 = torch.nn.Linear(3, 64)
        self.linear2 = torch.nn.Linear(64, 64)
        self.linear3 = torch.nn.Linear(64, 64)
        self.linear4 = torch.nn.Linear(64, outdim)

        self.activation = torch.nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        return x


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

def FDM_Wave(u, D=1, c=1.0):
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

def cLoss(u, u0, c=1.0, f_weights=None, ic_weights=None):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    boundary_u = u[:, 0, :, 0]
    u = u.reshape(batchsize, nt, nx)
    Du = FDM_Wave(u, c=c)[:, :, :]
    f = torch.zeros(Du.shape, device=u.device)
    
    ic_err = torch.mean(ic_weights * (u0[:, 0, :, 0] - boundary_u))
    f_err = torch.mean(f_weights * (Du - f))
    return ic_err, f_err

def L2(u, u0): 
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    boundary_u = u[:, 0, :, 0]
    u = u.reshape(batchsize, nt, nx)
    Du = FDM_Wave(u, c=1)[:, :, :]
    f = torch.zeros(Du.shape, device=u.device)
    
    ic_err = F.mse_loss(boundary_u, u0[:, 0, :, 0])
    f_err = F.mse_loss(Du, f)
    return ic_err, f_err

def data_loss(out, y):
    num_examples = x.size()[0]
    #Assume uniform mesh
    h = 1.0 / (x.size()[1] - 1.0)
    all_norms = h*torch.norm(out.view(num_examples,-1) - y.view(num_examples,-1), 2, 1)
    return torch.mean(all_norms)

parser = ArgumentParser(description='Basic paser')
parser.add_argument('--config_path', type=str, help='Path to the configuration file')
parser.add_argument('--log', action='store_true', help='Turn on the wandb')
args = parser.parse_args()
config_file = args.config_path
with open(config_file, 'r') as stream:
    config = yaml.load(stream, yaml.FullLoader)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_dataset = wave1Ddataset(
            config['train_data']['path'], 
            sub_x=config['train_data']['sub_x'],
            sub_t=config['train_data']['sub_t']
            )
train_loader = DataLoader(
            train_dataset, 
            batch_size=config['train_params']['batchsize'], 
            shuffle=False,
            )
# model_1_params = config['model_1']
# generator = FNN2d(
#         modes1=model_1_params['modes1'], 
#         modes2=model_1_params['modes2'], 
#         fc_dim=model_1_params['fc_dim'], 
#         layers=model_1_params['layers'],
#         activation=model_1_params['activation'], 
#     ).to(device)


# model_2_params = config['model_2']
# out_dim = 2
# discriminator = FNN2d(
#     modes1=model_2_params['modes1'], 
#     modes2=model_2_params['modes2'], 
#     fc_dim=model_2_params['fc_dim'], 
#     layers=model_2_params['layers'],
#     activation=model_2_params['activation'],
#     in_dim=3,
#     out_dim=out_dim,
# ).to(device)

generator = simpleLinear(outdim=1).to(device)
discriminator = simpleLinear(outdim=2).to(device)

# discriminator = simpleDiscriminator().to(device)
train_params = config['train_params']
optimizer = ACGD(max_params=discriminator.parameters(), 
                        min_params=generator.parameters(), 
                        lr_min=train_params['lr_min'], 
                        lr_max=train_params['lr_max'], 
                        tol=train_params["cg_tolerance"])

# optimizer_max = NAdam(discriminator.parameters(), betas=(0.9, 0.999),
#                 lr=config['train_params']['lr_max'])


# optimizer_min = Adam(generator.parameters(), betas=(0.9, 0.999),
#                 lr=config['train_params']['lr_min'])

generator.train()
discriminator.train()

epochs = config['train_params']['epochs']
pbar = tqdm(range(epochs), dynamic_ncols=True, smoothing=0.1)
ic_weight = config['train_params']['ic_loss']
f_weight = config['train_params']['f_loss']
xy_weight = config['train_params']['xy_loss']
guesses = []
for ep in pbar: 
    for x, y in train_loader: 
        optimizer.zero_grad()
        # optimizer_max.zero_grad()
        # optimizer_min.zero_grad()
        # x has shape [batchsize, Nt, Nx, (a, x, y)]
        x, y = x.to(device), y.to(device)            
        output = generator(x) 
        data_err = data_loss(output[:, 0, :, 0], y[:, 0, :, 0]) 
        bets = discriminator(x)
        ic_weights = bets[:, 0, :, 0]
        f_weights = bets[:, 0, :, 1]

        ic_err, f_err = cLoss(output, x, ic_weights=ic_weights, f_weights=f_weights)
        total_loss = f_weight * f_err + ic_weight * ic_err + xy_weight * data_err

        optimizer.step(loss=total_loss)
        # total_loss.backward()
        # optimizer_max.step()
        # optimizer_min.step()

        l2_ic_err, l2_f_err = L2(output, x)
        guesses.append([output[0, 0, :, 0].cpu().detach().numpy(), ic_weights.cpu().detach().numpy()])

    pbar.set_description(f"total_loss: {total_loss}, w_ic_err: {ic_err}, ic_err: {l2_ic_err}, ratio: {ic_weight * ic_err/xy_weight * data_err}")

fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(-0.25, 0.25))
line1, = ax.plot([], [], lw=3)
line2, = ax.plot([], [], lw=3)
line3, = ax.plot([], [], lw=3)
line1.set_label('ic guess')
line2.set_label('w guess')
line3.set_label('ic truth')
ax.legend()

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line2, line3

def animate(i, *args):
    ic_guess = args[0][i][0]
    w_guess = args[0][i][1]
    ic_truth = args[1]
    x = np.linspace(0, 1, 128)
    line1.set_data(x, ic_guess)
    line2.set_data(x, w_guess)
    line3.set_data(x, ic_truth)
    return line1,line2, line3

anim = FuncAnimation(fig, animate, init_func=init, fargs=(guesses, x[0, 0, :, 0].cpu().detach().numpy()),
                               frames=epochs, interval=30, blit=True)
anim.save('ic_guess_acgd.gif', writer='imagemagick')        

# python /groups/tensorlab/rgundaka/code/CPINO/cpino_run.py --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/base_configs/wave1D-cpino-test.yaml