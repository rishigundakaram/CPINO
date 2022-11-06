import torch

class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

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

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms
    
    def cLoss(self, x, y, weights):
        num_examples = x.size()[0]
        errs_w = torch.mean(weights.reshape(num_examples, -1) * (x.reshape(num_examples, -1)-y.reshape(num_examples, -1)))
        return errs_w
    
    
    
    def __call__(self, x, y):
        return self.rel(x, y)
    
def FDM_NS_vorticity(w, v=1/40, t_interval=1.0):
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

def PINO_loss3d(u, u0, forcing, v=1/40, t_interval=1.0):
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)

    u = u.reshape(batchsize, nx, ny, nt)
    lploss = LpLoss(size_average=True)

    u_in = u[:, :, :, 0]
    loss_ic = lploss(u_in, u0)

    Du = FDM_NS_vorticity(u, v, t_interval)
    f = forcing.repeat(batchsize, 1, 1, nt-2)
    loss_f = lploss(Du, f)

    return loss_ic, loss_f

def PINO_FDM(u, u0, forcing=None, v=1/40, t_interval=1.0, **kwargs):
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)

    u = u.reshape(batchsize, nx, ny, nt)
    u_in = u[:, :, :, 0]

    Du = FDM_NS_vorticity(u, v, t_interval)
    f = forcing.repeat(batchsize, 1, 1, nt-2)

    return u_in, u0, f, Du

class Loss():
    def __init__(self, config, physics_truth, p=2, **kwargs):
        self.model = config['info']['model']
        self.physics = physics_truth
        self.ic_weight = config['train_params']['ic_loss']
        self.f_weight = config['train_params']['f_loss']
        self.data_weight = config['train_params']['xy_loss']
        self.formulation = config['train_params']['loss_formulation']
        self.lploss = LpLoss()
        self.kwargs = kwargs
        self.p = p

    def physics_loss(self, u, u0): 
        ic_truth, ic_pred, f_truth, f_pred = self.physics(u, u0, **self.kwargs)
        ic_loss = self.lploss(ic_truth, ic_pred)
        f_loss = self.lploss(f_truth, f_pred) 
        return ic_loss, f_loss
    
    def w_physics_loss(self, u, u0, ic_weights, f_weights): 
        ic_truth, ic_pred, f_truth, f_pred = self.physics(u, u0, **self.kwargs)
        
        if self.model == "SAPINO" or self.model == "SAPINN": 
            ic_loss_w = self.lploss(ic_truth, ic_pred, weights=ic_weights)
        else: 
            ic_loss_w = self.lploss.cLoss(ic_truth, ic_pred, ic_weights)
        
        if self.model == "SAPINO" or self.model == "SAPINN": 
            f_loss_w = self.lploss(f_truth, f_pred, weights=f_weights)
        else: 
            f_loss_w = self.lploss.cLoss(f_truth, f_pred, f_weights)
        return ic_loss_w, f_loss_w
        
    def __call__(self, input, prediction, target=None):
        output = prediction["output"]
        pde_output = prediction["pde_output"]
        loss = {}
        data_loss = 0
        ic_loss = 0
        f_loss = 0
        
        ic_loss, f_loss = self.physics_loss(pde_output, input)
        loss["ic loss"] = ic_loss.item()
        loss["f loss"] = f_loss.item()
        
        if target is not None: 
            data_loss = self.lploss.rel(output, target)
            loss["data loss"] = data_loss.item()
        
        if "SA" in self.model or "C" in self.model:
            
            ic_weights = prediction["ic_weights"]
            f_weights = prediction["f_weights"]
            if self.formulation == 'competitive': 
                data_weights = prediction['data_weights']
                data_loss = self.cLoss(pde_output, target, weights=data_weights)
                loss["weighted data loss"] = data_loss.item()
            
            ic_loss, f_loss = self.w_physics_loss(
                pde_output, input, ic_weights, f_weights
                )
            loss["weighted f err"] =  f_loss.item()
            loss["weighted ic err"] = ic_loss.item()
            # loss["loss_x"] = self.ic_weight * ic_loss + self.f_weight * f_loss + self.data_weight * data_loss
            # loss["loss_y"] = self.ic_weight * ic_loss + self.f_weight * f_loss
            # return loss
        loss["loss"] = self.ic_weight * ic_loss + self.f_weight * f_loss + self.data_weight * data_loss
        return loss
    

class Mixed_Loss(Loss): 
    def __init__(self, config, physics_truth, p=2, **kwargs):
        super().__init__(config, physics_truth, p, **kwargs)
        self.mixed_loss = config['train_params']['mixed_loss']
        self.train_forcing = kwargs['train_forcing']
        self.tts_forcing = kwargs['tts_forcing']
    
    def __call__(self, obs_input, obs_target, obs_prediction, sampled_input, sampled_prediction):
        self.kwargs['forcing'] = self.kwargs['train_forcing']
        obs_loss = super().__call__(obs_input, obs_prediction, obs_target)
        self.kwargs['forcing'] = self.kwargs['tts_forcing']
        sampled_loss = super().__call__(sampled_input, sampled_prediction, target=None)
        loss = combine_dicts(obs_loss, sampled_loss, self.mixed_loss)
        return loss 

def combine_dicts(obs_loss, sample_loss, mixed_loss_const): 
    for key, value in obs_loss.items(): 
        if key in sample_loss: 
            obs_loss[key] = value + mixed_loss_const * sample_loss[key]
    return obs_loss