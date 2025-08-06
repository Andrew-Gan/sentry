#!/bin/bash

module --force purge
module load cuda gcc
source .venv/bin/activate

export TORCH_HOME=./torch

# Change to the directory from which you originally submitted this job.
cd $SLURM_SUBMIT_DIR

python agent_trainer.py --sig_out ./signatures --model_path ./torch private-key --private_key private.pem

deactivate
