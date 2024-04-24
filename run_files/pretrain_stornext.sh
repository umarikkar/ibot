#!/usr/bin/env bash
cd /vol/research/fmodel_medical/people/umar/ibot

set -e

/vol/research/fmodel_medical/people/umar/miniconda3/envs/ibot/bin/python -m torch.distributed.launch --nproc_per_node=$1 main_ibot.py --batch_size_per_gpu $2 --data_path '/vol/research/fmodel_medical/people/umar/datasets/tcga/tcga-coad/subimages/20x' --output_dir $3 --patch_size 16 --global_crops_number $4 --local_crops_number $5

