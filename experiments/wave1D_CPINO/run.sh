#!/bin/bash
#SBATCH --time=12:00:00  # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=64G   # memory per CPU core
#SBATCH --gres gpu:1
#SBATCH -J "wave1D_CPINO"  # job name
#SBATCH --mail-user=rgundaka@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/CPINO/run.py --log --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO/configs/wave1D-0.yaml &
wait
# python /groups/tensorlab/rgundaka/code/CPINO/run.py --config_path /groups/tensorlab/rgundaka/code/CPINO/experiments/wave1D_CPINO/configs/wave1D-0.yaml
