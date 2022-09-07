import itertools
import yaml
import os
import sys
from pprint import pprint

def linspace(a, b, num): 
    return [a + (a-b)/num * i for i in range(num+1)]

# experiment_file='CPINO_NS3D_Re500.yaml'
# walltime="43:00:00"
# experiment_name='CPINO_NS3D_Re500'

experiment_file='Wave1D.yaml'
walltime="01:10:00"
experiment_name='Wave1D'

base_dir='/groups/tensorlab/rgundaka/code/CPINO/'
experiment_dir='experiments/'
run_dir = 'runs'
config_file = os.path.join(base_dir, experiment_dir, 'experiment_configs/',experiment_file)


with open(config_file, 'r') as stream:
        config_file = yaml.load(stream, yaml.FullLoader)

if sys.argv[1] == 'wave1D':
    base_config = os.path.join(base_dir, experiment_dir, 'base_configs/wave1D.yaml')
elif sys.argv[1] == 'NS3D': 
    base_config = os.path.join(base_dir, experiment_dir, 'base_configs/NS3D.yaml')
elif sys.argv[1] == 'NS3D-100': 
    base_config = os.path.join(base_dir, experiment_dir, 'base_configs/Re100-05s.yaml')
else: 
    raise ValueError('invalid pde provided as an argument')


pde = sys.argv[1]

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

def create_sh(path, params, time="24:00:00", name="CPINO"): 
    n_tasks = len(params)
    for idx in range(n_tasks):
        cur_path = os.path.join(path, 'slurm', f'slurm_{idx}.sh')
        with open(cur_path, 'w+') as file: 
            file.write(
f"""#!/bin/bash
#SBATCH --time={time}  # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=64G   # memory per CPU core
#SBATCH --gres gpu:1
#SBATCH -J "{name}"  # job name
#SBATCH --mail-user=rgundaka@caltech.edu   # email address
#SBATCH --mail-type=FAIL"""
            )
            file.write('\n')
            train_str =  f'python {os.path.join(base_dir, "train.py")} --log --config_path {os.path.join(base_dir, experiment_dir, run_dir, experiment_name, f"configs/{pde}")}-{idx}.yaml'
            file.write(f"{train_str}\n")
            eval_str =  f'python {os.path.join(base_dir, "eval.py")} --config_path {os.path.join(base_dir, experiment_dir, run_dir, experiment_name, f"configs/{pde}")}-{idx}.yaml'
            file.write(f"{eval_str}\n")
    sh_path = os.path.join(path, 'run.sh')
    with open(sh_path, 'w+') as run: 
        for idx in range(n_tasks): 
            run.write(f'sbatch {os.path.join(path, "slurm", f"slurm_{idx}.sh")}\n')
    os.system(f'chmod +x {sh_path}')


if not os.path.exists(os.path.join(base_dir, experiment_dir, run_dir, experiment_name)):
    os.mkdir(os.path.join(base_dir, experiment_dir, run_dir, experiment_name))
if not os.path.exists(os.path.join(base_dir, experiment_dir, run_dir, experiment_name, 'configs')):
    os.mkdir(os.path.join(base_dir, experiment_dir, run_dir, experiment_name, 'configs'))
if not os.path.exists(os.path.join(base_dir, experiment_dir, run_dir, experiment_name, 'checkpoints')): 
    os.mkdir(os.path.join(base_dir, experiment_dir, run_dir, experiment_name, 'checkpoints'))
if not os.path.exists(os.path.join(base_dir, experiment_dir, run_dir, experiment_name, 'slurm')): 
    os.mkdir(os.path.join(base_dir, experiment_dir, run_dir, experiment_name, 'slurm'))

params = []
for param_grid in config_file: 
    params.extend(list(paths(param_grid)))
with open(base_config, 'r') as stream: 
    config = yaml.load(stream, yaml.FullLoader)

config['info']['save_dir'] = os.path.join(base_dir, experiment_dir, run_dir, experiment_name, 'checkpoints')
for idx, param in enumerate(params): 
    cur_config = update_config(config, param)
    cur_config['info']['project'] = f"{pde}"
    cur_config['info']['group'] = experiment_name
    cur_config['info']['save_name'] = f'{pde}-cpino-{idx}.pt'
    cur_path = os.path.join(base_dir, experiment_dir, run_dir, experiment_name)
    with open(os.path.join(cur_path, f'configs/{pde}-{idx}.yaml'), 'w') as outfile:
            yaml.dump(cur_config, outfile)
    

slurm_path = os.path.join(base_dir, experiment_dir, run_dir, experiment_name)
create_sh(slurm_path, params, time=walltime, name=experiment_name)    
# print(f'python {os.path.join(base_dir, "train_operator.py")} --log --config_path {os.path.join(base_dir, experiment_dir, "train/Darcy-train")}-{idx}.yaml')
# print(f'python {os.path.join(base_dir, "eval_operator.py")} --log --config_path {os.path.join(base_dir, experiment_dir, "train/Darcy-test")}-{idx}.yaml')




