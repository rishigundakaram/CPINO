from time import time
import yaml
import torch
from argparse import ArgumentParser
from data.problem import wave1D, Loss
import wandb
from tqdm import tqdm
import os

from model.CPINO import CPINO
from model.FNO import FNO
from pprint import pprint

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

def logger(dict, run=None, train=True): 
    new = {}
    for key, value in dict.items(): 
        if train: 
            new["train " + key] = value
        else:
            run.summary["test " + key] = value
    if train: 
        wandb.log(new)
    return None

if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    args = parser.parse_args()
    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    
    print('loading data')
    match config['info']['name']: 
        case "wave1D": 
            problem = wave1D(config)
    print('data is loaded')

    if config['info']['competitive']: 
        model = CPINO(config)
    else: model = FNO(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"using device: {device}")


    if args.log: 
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
    loss = Loss(config, problem.physics_truth)
    pbar = tqdm(range(epochs), dynamic_ncols=True, smoothing=0.1)
    for ep in pbar: 
        total_loss = {}
        for x, y in train_loader: 
            x, y = x.to(device), y.to(device)            
            output = model.predict(x) 
            cur_loss = loss(x, y, output)
            model.step(cur_loss)
            total_loss = update_loss_dict(total_loss, cur_loss)
        model.schedule_step()
        total_loss = loss_metrics(total_loss)
        if args.log: 
            logger(total_loss)
        pbar.set_description(dict_to_str(total_loss))
    total_loss = {}
    model.eval()
    for x, y in test_loader: 
        x, y = x.to(device), y.to(device)
        output = model.predict(x)
        cur_loss = loss(x, y, output)
        total_loss = update_loss_dict(total_loss, cur_loss)
    total_loss = loss_metrics(total_loss) 
    if args.log: 
        logger(total_loss, run, train=False)
    save_path = os.path.join(config['info']['save_dir'], config['info']['save_name'])
    model.save(save_path)
    print(total_loss)
        



