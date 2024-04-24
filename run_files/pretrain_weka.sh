#!/usr/bin/env bash
cd /mnt/fast/nobackup/users/um00109/umar/ibot

set -e

/mnt/fast/nobackup/users/um00109/umar/miniconda3/envs/ibot/bin/python -m torch.distributed.launch --nproc_per_node=$1 main_ibot.py --batch_size_per_gpu $2 --data_path '/mnt/fast/nobackup/scratch4weeks/um00109/tcga/tcga-coad/subimages/20x' --output_dir $3 --patch_size 16 --global_crops_number $4 --local_crops_number $5

