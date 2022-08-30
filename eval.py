from time import time
import yaml
import torch
from argparse import ArgumentParser
from train_utils.problem import wave1D, NS3D, Loss
import wandb
from tqdm import tqdm
import os

from model.Competitive import CPINO, CPINN, CPINO_SPLIT
from model.SAweights import SAPINN, SAPINO
from model.PINN import PINN
from model.PINO import PINO

from pprint import pprint
import matplotlib.pyplot as plt
from time import time

def update_loss_dict(total, cur): 
    cur["loss"] = cur["loss"].item()
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

def eval_loss(loader, model, loss): 
    total_loss = {}
    model.eval()
    for x, y in loader: 
        x, y = x.to(device), y.to(device)
        output = model.predict(x)
        cur_loss = loss(x, y, output)
        total_loss = update_loss_dict(total_loss, cur_loss)
    total_loss = loss_metrics(total_loss) 
    return total_loss

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

if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    args = parser.parse_args()
    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")

    print('loading data')
    match config['info']['name']: 
        case "wave1D": 
            problem = wave1D(config)
        case "NS":
            problem = NS3D(config)
    print('data is loaded')

    match config['info']['model']: 
        case "CPINO":
            model = CPINO(config)
        case "CPINO-split":
            model = CPINO_SPLIT(config)
        case "CPINN": 
            model = CPINN(config)
        case "SAPINO":
            model = SAPINO(config)
        case "SAPINN": 
            model = SAPINN(config)
        case "PINO":
            model = PINO(config)
        case "PINN":
            model = PINN(config)
        

    if args.log: 
        run = wandb.init(project=config['info']['project'],
                         entity=config['info']['entity'],
                         group=config['info']['group'],
                         config=config,
                         tags=config['info']['tags'], reinit=True,
                         settings=wandb.Settings(start_method="fork"))


    model.eval()
    epochs = config['train_params']['epochs']
    test_loader = problem.test_loader

    if config['info']['name'] == 'NS': 
        loss = Loss(config, problem.physics_truth, forcing=problem.test_forcing, 
            v=problem.v, t_interval=problem.test_t_interval)
    else: 
         loss = Loss(config, problem.physics_truth)
    pbar = tqdm(range(epochs), dynamic_ncols=True, smoothing=0.1)
    

    test_loss = eval_loss(problem.test_loader, model, loss)

    if args.log: 
        logger(test_loss, run, prefix='test')


