#!/bin/bash
#SBATCH -J resa_32
#SBATCH -o resa_32.out
#SBATCH -p rtxtt
#SBATCH --qos=rtx1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 1440
#SBATCH --gres=gpu:rtxtt:1

source activate laneatt

python main.py train --exp_name resa_32 --cfg cfgs/laneatt_tusimple_resnet34.yml

