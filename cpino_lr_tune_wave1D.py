from time import time
import yaml
import torch
from argparse import ArgumentParser
from train_utils.problem import wave1D, NS3D, Loss
import wandb
from tqdm import tqdm
import os

from model.Competitive import CPINO, CPINN
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

def config_update(args, config): 
    config['model_1']['activation'] = getattr(args, 'model_1.activation')
    config['model_2']['activation'] = getattr(args, 'model_2.activation')
    config['train_params']['batchsize'] = getattr(args, 'train_params.batchsize')
    config['train_params']['cg_tolerance'] = getattr(args, 'train_params.cg_tolerance')
    config['train_params']['lr_min'] = getattr(args, 'train_params.lr_min')
    config['train_params']['lr_max'] = getattr(args, 'train_params.lr_max')
    return config

def get_loss(loader, model, loss): 
    total_loss = {}
    model.eval()
    for x, y in loader: 
        x, y = x.to(device), y.to(device)
        output = model.predict(x)
        cur_loss = loss(x, y, output)
        total_loss = update_loss_dict(total_loss, cur_loss)
    total_loss = loss_metrics(total_loss) 
    return total_loss

if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--model_1.activation', type=str)
    parser.add_argument('--model_2.activation', type=str)
    parser.add_argument('--train_params.batchsize', type=int)
    parser.add_argument('--train_params.cg_tolerance', type=float)
    parser.add_argument('--train_params.lr_max', type=float)
    parser.add_argument('--train_params.lr_min', type=float)
    
    args = parser.parse_args()
    config_file = '/groups/tensorlab/rgundaka/code/CPINO/experiments/base_configs/wave1D.yaml'
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    
    config = config_update(args, config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    problem = wave1D(config)
    model = CPINO(config)

    run = wandb.init(project=config['info']['project'],
                        entity=config['info']['entity'],
                        group=config['info']['group'],
                        config=config,
                        tags=config['info']['tags'], reinit=True,
                        settings=wandb.Settings(start_method="fork"))


    model.train()
    epochs = config['train_params']['epochs']
    train_loader = problem.train_loader
    test_loader = problem.test_loader
    valid_loader = problem.valid_loader

    
    loss = Loss(config, problem.physics_truth)
    pbar = tqdm(range(epochs), dynamic_ncols=True, smoothing=0.1)
    start_time = time()
    runtime_min = config['info']['walltime']
    es_min = config['early_stopping']['minutes']
    es_start = start_time
    patience = config['early_stopping']['patience']
    cur_patience = patience
    delta = config['early_stopping']['delta']
    valid_error = None
    for ep in pbar: 
        total_loss = {}
        for idx, (x, y) in enumerate(train_loader): 
            x, y = x.to(device), y.to(device) 
            output = model.predict(x) 
            cur_loss = loss(x, y, output)
            model.step(cur_loss)
            total_loss = update_loss_dict(total_loss, cur_loss)
        model.schedule_step()
        total_loss = loss_metrics(total_loss)
        
        logger(total_loss)
        
        pbar.set_description(dict_to_str(total_loss))
        
        elapsed = time() - start_time
        if runtime_min is not None and elapsed > runtime_min * 60: 
            break

        valid_loss = get_loss(problem.valid_loader, model, loss)
        
        if valid_error is None: 
            valid_error = valid_loss['L2 data loss']
            print(valid_error)
        else: 
            if time() - es_start > es_min * 60: 
                cur_valid_error = valid_loss['L2 data loss']
                if cur_valid_error > valid_error or valid_error - cur_valid_error < delta: 
                    cur_patience -= 1
                    if cur_patience == 0: 
                        break
                else: 
                    cur_patience = patience
                es_start = time()
                valid_error = cur_valid_error
                print(valid_error)


    test_loss = get_loss(problem.test_loader, model, loss)

    logger(test_loss, run, prefix='test')
    save_path = os.path.join(config['info']['save_dir'], config['info']['save_name'])
    model.save(save_path)
        



