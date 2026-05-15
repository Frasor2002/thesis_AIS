#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -N 1

#SBATCH --job-name=xil_test
#SBATCH --output=./cluster/outputs2/output.out
#SBATCH --error=./cluster/outputs2/error.err
#SBATCH --partition edu-thesis
#SBATCH -t 1-00:00:0


source venv-pytorch/bin/activate

python "${1}.py"