#!/bin/bash

WANDB_USER="wandb-user" # wandb user (entity) -- change this to your wandb user

echo "Running lrc-ssm script"

nvidia-smi

# SelfRegulationSCP2 dataset parameters
NINPS=7 # scp2 has 7 input channels, don't change this
NCLASS=2 # scp2 has 2 classes, don't change this
NSEQUENCE=1152 # scp2 has 1152 sequence length, don't change this

# Run an individual training run with the following parameters
DATASET=scp2_2345 # dataset split to use, change this to scp2_3456, scp2_4567, scp2_5678, or scp2_6789 for other splits
LR=1e-4 # learning rate
INP_ENC=64 # input encoding dimension (hidden dimension)
NSTATE=64 # ssm states dimension
NLAYER=6 # number of layers/blocks
PATIENCE=1000 # early stopping patience

# Change conda environment name elk_py312 if needed
conda run -n elk_py312 python run_train.py --lrc_type lrc --precision 32 --datafile $DATASET --ninps $NINPS --nsequence $NSEQUENCE --nclass $NCLASS --lr $LR --ninp_enc $INP_ENC --nstates $NSTATE --nlayer $NLAYER --patience $PATIENCE --patience_metric accuracy --quasi --wandb_user $WANDB_USER


