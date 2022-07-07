#/bin/bash

# set up env
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38

# train a GVP-GNN model
python lm-gvp/train.py \
    --dataset_dir protein_data \
    --task cc \
    --model_name gvp \
    --gpus 1 \
    --bs 64 \
    --lr 1e-3 \
    --max_epochs 10 \
    --early_stopping_patience 5 \
    --accelerator gpu

# train a LM-GVP model
python lm-gvp/train.py \
    --dataset_dir protein_data \
    --task cc \
    --model_name bert_gvp \
    --freeze_bert True \
    --gpus 1 \
    --bs 32 \
    --lr 1e-4 \
    --max_epochs 10 \
    --early_stopping_patience 5 \
    --accelerator gpu
