#!/bin/bash

nvcc -Xcompiler '-fPIC' -o ./RapidEC/gsv.so -shared ./RapidEC/gsv.cu

# Change to the directory from which you originally submitted this job.
cd $SLURM_SUBMIT_DIR

python agent_trainer.py --sig_out ./signatures --model_path ./torch private-key --private_key private.pem

rm ./RapidEC/gsv.so
