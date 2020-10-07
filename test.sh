#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=p40_4,p100_4,v100_sxm2_4,dgx1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=2:00:00
#SBATCH --array=0
#SBATCH --job-name=test
#SBATCH --output=test_%A_%a.out

module purge
module load cuda/10.1.105

python -u /home/eo41/complex-neurons/test.py --arity 3

echo "Done"
