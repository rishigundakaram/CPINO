import itertools
import yaml
import os
import sys


param_grid = {
    'info' : {
        'competitive': [True]
    }
}
walltime="12:00:00"
experiment_name='wave1D_CPINO'

base_dir='/groups/tensorlab/rgundaka/code/CPINO/'
experiment_dir='experiments'

if sys.argv[1] == 'wave1D':
    base_config = os.path.join(base_dir, experiment_dir, 'base_configs/wave1D.yaml')
    base_config_test = os.path.join(base_dir, 'CPINO/base_configs/Darcy-test.yaml')
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
            train_str =  f'python {os.path.join(base_dir, "run.py")} --log --config_path {os.path.join(base_dir, experiment_dir, experiment_name, f"configs/{pde}")}-{idx}.yaml'
            file.write(f"srun -n 1 --nodes=1 {train_str} &\n")
        file.write('wait\n')


if not os.path.exists(os.path.join(base_dir, experiment_dir, experiment_name)):
    os.mkdir(os.path.join(base_dir, experiment_dir, experiment_name))
if not os.path.exists(os.path.join(base_dir, experiment_dir, experiment_name, 'configs')):
    os.mkdir(os.path.join(base_dir, experiment_dir, experiment_name, 'configs'))
if not os.path.exists(os.path.join(base_dir, experiment_dir, experiment_name, 'checkpoints')): 
    os.mkdir(os.path.join(base_dir, experiment_dir, experiment_name, 'checkpoints'))


params = list(paths(param_grid))

with open(base_config, 'r') as stream: 
    config = yaml.load(stream, yaml.FullLoader)

config['info']['save_dir'] = os.path.join(base_dir, experiment_dir, experiment_name, 'checkpoints')
for idx, param in enumerate(params): 
    cur_config = update_config(config, param)
    cur_config['info']['project'] = f"CPINO-{pde}"
    cur_config['info']['group'] = [experiment_name]
    cur_config['info']['save_name'] = f'{pde}-cpino-{idx}.pt'
    cur_path = os.path.join(base_dir, experiment_dir, experiment_name)
    with open(os.path.join(cur_path, f'configs/{pde}-{idx}.yaml'), 'w') as outfile:
            yaml.dump(cur_config, outfile)
    

slurm_path = os.path.join(base_dir, experiment_dir, experiment_name, 'run.sh')
create_sh(slurm_path, params, nodes=len(params), time=walltime, name=experiment_name)    
# print(f'python {os.path.join(base_dir, "train_operator.py")} --log --config_path {os.path.join(base_dir, experiment_dir, "train/Darcy-train")}-{idx}.yaml')
# print(f'python {os.path.join(base_dir, "eval_operator.py")} --log --config_path {os.path.join(base_dir, experiment_dir, "train/Darcy-test")}-{idx}.yaml')




