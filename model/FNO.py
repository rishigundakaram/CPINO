import torch
import torch.nn as nn
import torch.nn.functional as F



from train_utils.adam import Adam

from .basics import SpectralConv2d, Model


class FNO(Model): 
    def __init__(self, params) -> None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cur_params = params['model_1']
        self.model =  FNN2d(
            modes1=cur_params['modes1'], 
            modes2=cur_params['modes2'], 
            fc_dim=cur_params['fc_dim'], 
            layers=cur_params['layers'],
            activation=cur_params['activation'], 
        ).to(device)
        self.ic_weight = params["train_params"]["ic_loss"]
        self.data_weight = params["train_params"]["xy_loss"]
        self.f_weight = params["train_params"]["f_loss"]

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

        

class FNN2d(nn.Module):
    def __init__(self, modes1, modes2,
                 width=64, fc_dim=128,
                 layers=None,
                 in_dim=3, out_dim=1,
                 activation='tanh'):
        super(FNN2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2d(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        if activation =='tanh':
            self.activation = F.tanh
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation == F.relu
        else:
            raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        '''
        Args:
            - x : (batch size, x_grid, y_grid, 2)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y)
            x = x1 + x2
            if i != length - 1:
                x = self.activation(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
