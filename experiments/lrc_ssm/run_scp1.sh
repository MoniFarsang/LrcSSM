#!/bin/bash

WANDB_USER="wandb-user" # wandb user (entity) -- change this to your wandb user

echo "Running lrc-ssm script"

nvidia-smi

# SelfRegulationSCP1 dataset parameters
NINPS=6 # scp1 has 6 input channels, don't change this
NCLASS=2 # scp1 has 2 classes, don't change this
NSEQUENCE=896 # scp1 has 896 sequence length, don't change this

# Run an individual training run with the following parameters
DATASET=scp1_2345 # dataset split to use, change this to scp1_3456, scp1_4567, scp1_5678, or scp1_6789 for other splits
LR=1e-4 # learning rate
INP_ENC=64 # input encoding dimension (hidden dimension)
NSTATE=64 # ssm states dimension
NLAYER=6 # number of layers/blocks
PATIENCE=1000 # early stopping patience

# Change conda environment name elk_py312 if needed
conda run -n elk_py312 python run_train.py --lrc_type lrc --datafile $DATASET --ninps $NINPS --nsequence $NSEQUENCE --nclass $NCLASS --lr $LR --ninp_enc $INP_ENC --nstates $NSTATE --nlayer $NLAYER --patience $PATIENCE --patience_metric accuracy --quasi --wandb_user $WANDB_USER

