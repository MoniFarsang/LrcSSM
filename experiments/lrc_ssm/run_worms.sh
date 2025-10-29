#!/bin/bash

WANDB_USER="wandb-user" # wandb user (entity) -- change this to your wandb user

echo "Running lrc-ssm script"

nvidia-smi

#Eigenworms dataset parameters
NINPS=6 # eigenworms has 6 input channels, don't change this
NCLASS=5 # eigenworms has 5 classes, don't change this
NSEQUENCE=17984 # eigenworms has 17984 sequence length, don't change this

# Run an individual training run with the following parameters
DATASET=eigenworms_2345 # dataset split to use, change this to eigenworms_3456, eigenworms_4567, eigenworms_5678, or eigenworms_6789 for other splits
LR=1e-4 # learning rate
INP_ENC=64 # input encoding dimension (hidden dimension)
NSTATE=64 # ssm states dimension
NLAYER=6 # number of layers/blocks
PATIENCE=100 # early stopping patience

# Change conda environment name elk_py312 if needed
conda run -n elk_py312 python run_train.py --lrc_type lrc --datafile $DATASET --ninps $NINPS --nsequence $NSEQUENCE --nclass $NCLASS --lr $LR --ninp_enc $INP_ENC --nstates $NSTATE --nlayer $NLAYER --patience $PATIENCE --patience_metric accuracy --quasi --wandb_user $WANDB_USER

