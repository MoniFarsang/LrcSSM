#!/bin/bash

WANDB_USER="wandb-user" # wandb user (entity) -- change this to your wandb user

echo "Running lrc-ssm script"

nvidia-smi

# MotorImagery dataset parameters
NINPS=63 # motor has 63 input channels, don't change this
NCLASS=2 # motor has 2 classes, don't change this
NSEQUENCE=3000 # motor has 3000 sequence length, don't change this

# Run an individual training run with the following parameters
DATASET=motor_2345 # dataset split to use, change this to motor_3456, motor_4567, motor_5678, or motor_6789 for other splits
LR=1e-4 # learning rate
INP_ENC=64 # input encoding dimension (hidden dimension)
NSTATE=64 # ssm states dimension
NLAYER=6 # number of layers/blocks
PATIENCE=1000 # early stopping patience

# Change conda environment name elk_py312 if needed
conda run -n elk_py312 python run_train.py --lrc_type lrc --datafile $DATASET --ninps $NINPS --nsequence $NSEQUENCE --nclass $NCLASS --lr $LR --ninp_enc $INP_ENC --nstates $NSTATE --nlayer $NLAYER --patience $PATIENCE --patience_metric accuracy --quasi --wandb_user $WANDB_USER

