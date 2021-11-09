#!/bin/bash
#SBATCH -J check
#SBATCH -o check.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 2880
#SBATCH --gres=gpu:ttxp:1

source activate laneatt

python check.py