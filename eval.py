from time import time
import yaml
import torch
from argparse import ArgumentParser
import wandb
import os

from pprint import pprint
from time import time

from train import setup_loss, setup_model, setup_problem, eval_loss

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
    problem = setup_problem(config)
    print('data is loaded')

    
        

    name = config['info']['save_name']
    name = name.split('.')
    id = name[0].split('-')[-1]
    api = wandb.Api()
    entity = config['info']['entity']
    project = config['info']['project']
    run = api.run(f'{entity}/{project}/{id}')
        

    save_path = os.path.join(config['info']['save_dir'], config['info']['save_name'])
    model = setup_model(config)
    model.load(save_path)
    
    model.eval()
    test_loader = problem.val_loader

    loss = setup_loss(config, problem)  

    test_loss = eval_loss(test_loader, model)
    print(test_loss)
    
    logger(test_loss, run, prefix='test')

    run.update()