from time import time
import yaml
import torch
from argparse import ArgumentParser
import wandb
from tqdm import tqdm
import os

from model.Competitive import CPINO, CPINN, CPINO_SPLIT
from model.SAweights import SAPINN, SAPINO
from model.PINN import PINN
from model.PINO import PINO

from train_utils.problem import wave1D, NS3D, KF
from train_utils.loss import Loss, Mixed_Loss, LpLoss, PINO_loss3d, PINO_FDM
from train_utils.schedulers import decay_schedule, no_schedule
from train_utils.utils import get_forcing, sample_data


from time import time
import numpy as np

import atexit


def update_loss_dict(total, cur): 
    cur["loss"] = cur["loss"].item()
    if "loss_x" in cur.keys(): 
        cur["loss_x"] = cur["loss_x"].item()
        cur["loss_y"] = cur["loss_y"].item()
    if not total: 
        
        cur['batches'] = 1
        return cur
    
    for key, val in cur.items(): 
        total[key] += val
    total['batches'] += 1
    return total

def loss_metrics(total): 
    batches = total['batches']
    for key, val in total.items(): 
        total[key] = val/batches
    del total['batches']
    return total

def dict_to_str(dict): 
    str = ""
    for key, value in dict.items(): 
        if '_' in key: 
            continue
        str += f"{key}: {value:.5f} "
    return str

def logger(dict, run=None, prefix='train'): 
    new = {}
    for key, value in dict.items(): 
        new[prefix + ' ' + key] = value
        if prefix == 'test': 
            run.summary[prefix + ' ' + key] = value
    if prefix != 'test': 
        wandb.log(new)
    return None


def eval_loss(loader, model): 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    lploss = LpLoss()
    val_err = []
    for u, a in loader: 
        u, a = u.to(device), a.to(device)
        output = model(a)
        
        val_err.append(lploss(output, u).item())
    
    N = len(loader)
    avg_err = np.mean(val_err)
    std_err = np.std(val_err, ddof=1) / np.sqrt(N)
    return {"loss": avg_err, "std err loss": std_err}

def check_early_stopping(prev_metric, cur_metric, cur_epoch, min_epochs, cur_patience, patience, delta): 
    flag = 0
    if prev_metric is None: 
        pass
    elif cur_epoch > min_epochs and (cur_metric > prev_metric or prev_metric - cur_metric > delta): 
        cur_patience -= 1
        if cur_patience == 0: 
            flag = 1
    else: 
        cur_patience = patience
    return flag, cur_metric, cur_patience 

def end(save_path, model): 
        model.save(save_path)

def train(config, device, model, u_loader, a_loader, forcing):
    total_loss = {}
    ic_weight = config['train_params']['ic_loss']
    f_weight = config['train_params']['f_loss']
    xy_weight = config['train_params']['xy_loss']
    lploss = lploss = LpLoss(size_average=True)
    model_type = config['info']['model']
    
    v = 1/ config['data']['Re']
    t_duration = config['data']['t_duration']
    if xy_weight > 0:
            u, a_in = next(u_loader)
            u = u.to(device)
            a_in = a_in.to(device)
            out = model.predict(a_in)
            data_loss = lploss(out["output"], u)
            total_loss["data loss"] = data_loss.item()
            if "C" in model_type and config['train_params']['loss_formulation'] == 'competitive': 
                data_loss = lploss.cLoss(out["output"], u, weights=out["data_weights"])
                total_loss["weighted data loss"] = data_loss.item()
    else:
        data_loss = torch.zeros(1, device=device)

    if f_weight != 0.0:
        # pde loss
        a = next(a_loader)
        a = a.to(device)
        out = model.predict(a)
        
        u0  = a[:, :, :, 0, -1]
        ic_loss, f_loss = PINO_loss3d(out["output"], u0, forcing, v, t_duration)
        total_loss['ic loss'] = ic_loss.item()
        total_loss['f loss'] = f_loss.item()
        if "C" in model_type: 
            ic_truth, ic_pred, f_truth, f_pred = PINO_FDM(out["output"], u0, forcing, v, t_duration)
            ic_loss = lploss.cLoss(ic_truth, ic_pred, out["ic_weights"])
            f_loss = lploss.cLoss(f_truth, f_pred, out["f_weights"])
            total_loss["weighted f err"] =  f_loss.item()
            total_loss["weighted ic err"] = ic_loss.item()
    else:
        ic_loss = f_loss = 0.0

    total_loss["loss"] = data_loss * xy_weight + f_loss * f_weight + ic_loss * ic_weight
    print(total_loss)
    model.step(total_loss)
    return total_loss

# def train(update_loss_dict, loss_metrics, device, model, u_loader, a_loader, loss):
#     total_loss = {}
#     for (u, a_in), a in zip(u_loader, a_loader): 
#         s = time()
#         # print(f"before iter: {torch.cuda.memory_allocated(device) / 1e9}")
#         a_in, u, a = a_in.to(device), u.to(device), a.to(device)
#         output = model.predict(a_in, a)
#         # print(f"after pred: {torch.cuda.memory_allocated(device) / 1e9}") 
#         cur_loss = loss(a, output, u)
#         # print(f"after loss: {torch.cuda.memory_allocated(device) / 1e9}")
#         model.step(cur_loss)
#         # print(f"after step: {torch.cuda.memory_allocated(device) / 1e9}")
#         e = time()
#         # print(f"estimated time: {(e-s)/60*400}")
#         total_loss = update_loss_dict(total_loss, cur_loss)
#     total_loss = loss_metrics(total_loss)
#     return total_loss

def mixed_train(update_loss_dict, loss_metrics, device, model, train_loader, tts_loader, loss):
    total_loss = {}
    for x, y in train_loader: 
        obs_x, obs_y = x.to(device), y.to(device)
        obs_output = model.predict(obs_x) 
        tts_x = next(tts_loader)
        tts_x = tts_x.to(device)
        tts_output = model.predict(tts_x)
        cur_loss = loss(obs_x, obs_y, obs_output, tts_x, tts_output)
        print(cur_loss)
        model.step(cur_loss)
        total_loss = update_loss_dict(total_loss, cur_loss)
        print('step')
    total_loss = loss_metrics(total_loss)
    return total_loss


def setup_problem(config):
    if config['info']['name'] == "wave1D": 
        problem = wave1D(config)
    elif config['info']['name'] == "NS":
        problem = NS3D(config)
    elif config['info']['name'] == "KF":
        problem = KF(config)
    return problem

def setup_model(config):
    if config['info']['model'] == "CPINO": 
        model = CPINO(config)
    elif config['info']['model'] == "CPINO-split": 
        model = CPINO_SPLIT(config)
    elif config['info']['model'] == "CPINN": 
        model = CPINN(config)
    elif config['info']['model'] == "SAPINO": 
        model = SAPINO(config)
    elif config['info']['model'] == "SAPINN": 
        model = SAPINN(config)
    elif config['info']['model'] == "PINO": 
        model = PINO(config)
    elif config['info']['model'] == "PINN": 
        model = PINN(config)
    return model

def setup_loss(config, problem):
    if config['info']['name'] == 'NS' or config['info']['name'] == 'KF': 
        if config['train_params']['tts_batchsize'] > 0: 
            loss = Mixed_Loss(config, problem.physics_truth, train_forcing=problem.train_forcing, 
                                tts_forcing=problem.tts_forcing, v=problem.v, 
                                t_interval=problem.train_t_interval)
        else: 
            loss = Loss(config, problem.physics_truth, forcing=problem.forcing, 
                v=problem.v, t_interval=problem.train_t_interval, 
                )
    else: 
         loss = Loss(config, problem.physics_truth)
    return loss

def main(config, problem, log=False): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    print('loading model')
    model = setup_model(config)
    num_params = model.count_params()
    config['num_params'] = num_params
    print(f'Number of parameters: {num_params}')
    
    if args.ckpt:
        ckpt_path = args.ckpt
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
        

    if log: 
        run = wandb.init(project=config['info']['project'],
                         entity=config['info']['entity'],
                         group=config['info']['group'],
                         config=config,
                         reinit=True,
                         settings=wandb.Settings(start_method="fork"))
        config['info']['save_name'] = run.name + '-' + run.id
        with open(config_file, 'w') as cf:
            yaml.dump(config, cf)

    loss = setup_loss(config, problem)
    
    model.train()
    epochs = config['train_params']['epochs']
    a_loader = sample_data(problem.a_loader)
    u_loader = sample_data(problem.u_loader)
    val_loader = problem.val_loader
    if config['early_stopping']['use']:   
        patience = config['early_stopping']['patience']
        min_epochs = config['early_stopping']['min_epochs']
        delta = config['early_stopping']['delta']
        prev_metric = None
        cur_patience = patience
    if config['train_params']['tts_batchsize'] > 0:
        tts_loader = problem.tts_loader   
    
    S = config['data']['pde_res'][0]
    forcing = get_forcing(S).to(device)
    
    pbar = tqdm(range(epochs), dynamic_ncols=True, smoothing=0.1)
    start_time = time()
    runtime_min = config['info']['walltime']
    tts_batchsize = config['train_params']['tts_batchsize']
    save_path = os.path.join(config['info']['save_dir'], config['info']['save_name'])
    # saves model when script finishes
    atexit.register(end, save_path=save_path, model=model)
    eval_step = config["train_params"]['eval_step']
    for ep in pbar: 
        if tts_batchsize == 0: 
            total_loss = train(config, device, model, u_loader, a_loader, forcing)
            print(total_loss)
        else: 
            raise (ValueError)        
        # elif tts_batchsize > 0: 
            # total_loss = mixed_train(update_loss_dict, loss_metrics, device, model, train_loader, tts_loader, loss)
        if log: 
            logger(total_loss, prefix='train')
        pbar.set_description(dict_to_str(total_loss))
        elapsed = time() - start_time
        if runtime_min != 0 and elapsed > runtime_min * 60: 
            break
        
        if config['data']['n_test_samples'] > 0 and ep % eval_step == 0:
            model.eval()
            valid_loss = eval_loss(val_loader, model)
            model.train()
            if log:
                logger(valid_loss, prefix='valid')

            if config['early_stopping']['use']:    
                cur_metric = valid_loss['data loss']
                flag, prev_metric, cur_patience = check_early_stopping(
                    prev_metric, cur_metric, 
                    ep, min_epochs, 
                    cur_patience, patience, 
                    delta)
                if flag: 
                    break

if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--test', action='store_true', help='Test')
    parser.add_argument('--tqdm', action='store_true', help='Turn on the tqdm')
    args = parser.parse_args()
    config_file = args.config
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    print('loading data')
    problem = setup_problem(config)
    main(config, problem, args.log)

    
        



