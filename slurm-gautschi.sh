#!/bin/bash

# Init
# module load conda cuda
# conda create --name sentry python=3.10
# conda activate sentry
# pip install -r requirements.txt
# nvcc -Xcompiler '-fPIC' -o ./RapidEC/gsv.so -shared ./RapidEC/gsv.cu
# python agent_dataset.py uoft-cs/cifar10 16 1 /scratch/gautschi/gan35/dataset/cifar10

# Run
# module --force purge
# module load conda cuda
# conda activate sentry
# sbatch --nodes=1 --ntasks=64 --gpus-per-node=1 -p smallgpu -a zghodsi --qos=normal slurm-gautschi.sh

cd $SLURM_SUBMIT_DIR
python agent_trainer.py --sig_out ./signatures --model_path ./torch private-key --private_key private.pem
python agent_inferencer.py --sig_path ./signatures --model_path /home/torch private-key --public_key public.pem

# End
# conda deactivate
