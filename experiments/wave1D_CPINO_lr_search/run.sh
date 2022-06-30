#!/bin/bash
#SBATCH --time=12:00:00  # walltime
#SBATCH --ntasks=25   # number of processor cores (i.e. tasks)
#SBATCH --nodes=25   # number of nodes
#SBATCH --mem=64G   # memory per CPU core
#SBATCH --gres gpu:1
#SBATCH -J "wave1D_CPINO_lr_search"  # job name
#SBATCH --mail-user=rgundaka@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-0.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-1.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-2.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-3.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-4.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-5.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-6.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-7.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-8.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-9.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-10.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-11.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-12.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-13.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-14.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-15.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-16.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-17.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-18.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-19.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-20.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-21.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-22.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-23.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO_lr_search/configs/wave1D-24.yaml &
wait
