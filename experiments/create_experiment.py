import itertools
import yaml
import os
import sys
from pprint import pprint

# param_grid = {
#     'train': {
#         'lr_min': [.05, .025, .01, .005, .001], 
#         'lr_max': [.05, .025, .01, .005, .001],
#         'epochs': [25], 
#         'batchsize': [60]
#     }
# }
# walltime="18:00:00"
# experiment_name='Darcy_cpino_lr'

# param_grid = {
#     'train': {
#         'lr_min': [.01], 
#         'lr_max': [.01],
#         'epochs': [25],
#         'batchsize': [20, 40, 60, 80, 100, 120]
#     }
# }
# walltime="18:00:00"
# experiment_name='Darcy_cpino_batchsize'

# param_grid = {
#     'model': {
#         'competitive': [True],
#         'competitive_input': [['FNN output', 'initial_conditions']]
#     }, 
#     'train': {
#         'batchsize': [60],
#         'epochs': [50],
#         'lr_min': [.005],
#         'lr_max': [.025],
#         'cg_tolerance': [10e-4]
#     }
# }
# walltime="18:00:00"
# experiment_name='Darcy_cpino'

# param_grid = {
#     'model': {
#         'competitive_input': [['FNN output'], ['FNN output', 'initial_conditions']]
#     }, 
#     'train': {
#         'batchsize': [20],
#         'epochs': [10],
#         'lr_min': [.005],
#         'lr_max': [.025], 
#         'cg_tolerance': [10e-4]
#     }
# }
# walltime="12:00:00"
# experiment_name='Darcy_cpino_input'

# param_grid = {
#     'model': {
#         'competitive_input': [['FNN output', 'initial_conditions']]
#     }, 
#     'train': {
#         'batchsize': [20],
#         'epochs': [40],
#         'lr_min': [.005],
#         'lr_max': [.025], 
#         'cg_tolerance': [10e-4, 10e-5, 10e-6, 10e-7, 10e-8, 10e-9, 10e-10]
#     }
# }
# walltime="24:00:00"
# experiment_name='Darcy_cpino_tolerance'

param_grid = {
    'model': {
        'competitive_input': [['FNN output', 'initial conditions'], ['initial conditions']]
    }, 
    'train': {
        'batchsize': [1],
        'epochs': [1],
        'lr_min': [.005],
        'lr_max': [.025], 
        'cg_tolerance': [.001]
    }
}
walltime="1:00:00"
experiment_name='NS_initial_test'

base_dir='/groups/tensorlab/rgundaka/code/PINO/'
experiment_dir='CPINO/experiments'


if sys.argv[1] == 'Darcy':
    base_config_train = os.path.join(base_dir, 'CPINO/base_configs/Darcy-train.yaml')
    base_config_test = os.path.join(base_dir, 'CPINO/base_configs/Darcy-test.yaml')
    pde = 'Darcy'
elif sys.argv[1] == 'NS':
    base_config_train = os.path.join(base_dir, 'CPINO/base_configs/NS-1s-train.yaml')
    base_config_test = os.path.join(base_dir, 'CPINO/base_configs/NS-05s-test.yaml')
    pde = 'NS'
else: 
    raise ValueError('invalid pde provided as an argument')


def paths(cur_dict):
    all_paths = []
    for key, value in cur_dict.items(): 
        if type(value) == list: 
            all_paths.append([{key: val} for val in cur_dict[key]])
        else:
            found = paths(cur_dict[key])
            all_paths.append([{key: i} for i in found])
    all_paths = itertools.product(*all_paths)
    all_params = []
    for i in all_paths: 
        cur = {}
        for j in i: 
            cur.update(j)
        all_params.append(cur)
    return all_params

def update_config(config, params): 
    for key, val in params.items(): 
        if type(val) != dict: 
            config[key] = val
        else: 
            update_config(config[key], params[key])
    return config

def create_sh(path, params, nodes=1, time="24:00:00", name="CPINO"): 
    n_tasks = len(params)
    with open(path, 'w+') as file: 
        file.write(
f"""#!/bin/bash
#SBATCH --time={time}  # walltime
#SBATCH --ntasks={n_tasks}   # number of processor cores (i.e. tasks)
#SBATCH --nodes={nodes}   # number of nodes
#SBATCH --mem=64G   # memory per CPU core
#SBATCH --gres gpu:1
#SBATCH -J "{name}"  # job name
#SBATCH --mail-user=rgundaka@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
"""
        )
        for idx in range(n_tasks):
            train_str =  f'python {os.path.join(base_dir, "train_operator.py")} --log --config_path {os.path.join(base_dir, experiment_dir, experiment_name, f"configs/train/{pde}-train")}-{idx}.yaml'
            file.write(f"srun -n 1 --nodes=1 {train_str} &\n")
        file.write('wait\n')
        for idx in range(n_tasks):
            test_str = f'python {os.path.join(base_dir, "eval_operator.py")} --log --config_path {os.path.join(base_dir, experiment_dir, experiment_name, f"configs/test/{pde}-test")}-{idx}.yaml'
            file.write(f"srun -n 1 --nodes=1 {test_str} &\n")
        file.write('wait\n')


if not os.path.exists(os.path.join(base_dir, experiment_dir, experiment_name)):
    os.mkdir(os.path.join(base_dir, experiment_dir, experiment_name))
if not os.path.exists(os.path.join(base_dir, experiment_dir, experiment_name, 'configs')):
    os.mkdir(os.path.join(base_dir, experiment_dir, experiment_name, 'configs'))
if not os.path.exists(os.path.join(base_dir, experiment_dir, experiment_name, 'configs/train')):
    os.mkdir(os.path.join(base_dir, experiment_dir, experiment_name, 'configs/train'))
if not os.path.exists(os.path.join(base_dir, experiment_dir, experiment_name, 'configs/test')):
    os.mkdir(os.path.join(base_dir, experiment_dir, experiment_name, 'configs/test'))



params = list(paths(param_grid))

with open(base_config_train, 'r') as stream: 
    config_train = yaml.load(stream, yaml.FullLoader)

with open(base_config_test, 'r') as stream: 
    config_test = yaml.load(stream, yaml.FullLoader)

for idx, param in enumerate(params): 
    cur_train_config = update_config(config_train, param)
    cur_train_config['train']['save_name'] = f'{pde}-cpino-{idx}.pt'
    cur_path = os.path.join(base_dir, experiment_dir, experiment_name)
    with open(os.path.join(cur_path, f'configs/train/{pde}-train-{idx}.yaml'), 'w') as outfile:
            yaml.dump(cur_train_config, outfile)

    cur_test_config = update_config(config_test, param)
    cur_test_config['test']['ckpt'] = os.path.join(cur_path, f'checkpoints/{pde}-cpino-{idx}.pt')
    with open(os.path.join(cur_path, f'configs/test/{pde}-test-{idx}.yaml'), 'w') as outfile:
            yaml.dump(cur_test_config, outfile)
    

slurm_path = os.path.join(base_dir, experiment_dir, experiment_name, 'run.sh')
create_sh(slurm_path, params, nodes=len(params), time=walltime, name=experiment_name)    
# print(f'python {os.path.join(base_dir, "train_operator.py")} --log --config_path {os.path.join(base_dir, experiment_dir, "train/Darcy-train")}-{idx}.yaml')
# print(f'python {os.path.join(base_dir, "eval_operator.py")} --log --config_path {os.path.join(base_dir, experiment_dir, "train/Darcy-test")}-{idx}.yaml')




