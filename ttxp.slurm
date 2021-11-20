#!/bin/bash
#SBATCH -J ttxp
#SBATCH -o ttxp.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 2880
#SBATCH --gres=gpu:ttxp:1

source activate laneatt1

python main.py train --exp_name r34_trans_128_001 --cfg cfgs/laneatt_tusimple_resnet34_trans_128_001.yml
