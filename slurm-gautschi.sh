#!/bin/bash

# Init
# conda create --name sentry python=3.10
# conda activate sentry
# pip install -r requirements.txt

# conda config --set channel_priority flexible 
# conda install nvidia/label/cuda-12.6.0::cuda-toolkit

# Run
# conda activate sentry
# sbatch --nodes=1 --ntasks=14 --gpus-per-node=1 --partition=ai --account=zghodsi --qos=normal slurm-gautschi.sh

cd $SLURM_SUBMIT_DIR
nvcc -Xcompiler '-fPIC' -o ./RapidEC/gsv.so -shared ./RapidEC/gsv.cu
python agent_trainer.py --sig_out ./signatures --model_path ./torch private-key --private_key private.pem
rm ./RapidEC/gsv.so

# End
# conda deactivate
