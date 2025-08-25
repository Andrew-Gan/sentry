#!/bin/bash

# Init
# module --force purge
# module load cuda gcc
# python -m venv .venv
# pip install -r requirements.txt
# nvcc -Xcompiler '-fPIC' -o ./RapidEC/gsv.so -shared ./RapidEC/gsv.cu
# python agent_dataset.py uoft-cs/cifar10 16 1 ./dataset/cifar10

# Run
# module --force purge
# module load cuda gcc
# source .venv/bin/activate
# sbatch -A zghodsi -p a30 --mem=8G --gres=gpu:1 slurm-gilbreth.sh

export TORCH_HOME=./torch

# Change to the directory from which you originally submitted this job.
cd $SLURM_SUBMIT_DIR

python agent_trainer.py --sig_out ./signatures --model_path ./torch private-key --private_key private.pem
python agent_inferencer.py --sig_path ./signatures --model_path ./torch private-key --public_key public.pem

# deactivate
