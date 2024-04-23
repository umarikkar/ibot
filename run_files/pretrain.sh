#!/usr/bin/env bash
cd /vol/research/fmodel_medical/people/umar/ibot

set -e

source /vol/research/fmodel_medical/people/umar/miniconda3/etc/profile.d/conda.sh
conda activate ibot

python -m torch.distributed.launch --nproc_per_node=4 main_ibot.py --batch_size_per_gpu 48 --data_path '/vol/research/fmodel_medical/people/umar/datasets/tcga/tcga-coad/subimages/20x/' --output_dir './outputs_patch16_2_10/' --patch_size 16 --global_crops_number 2 --local_crops_number 10

