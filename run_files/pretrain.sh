#!/usr/bin/env bash
cd /vol/research/fmodel_medical/people/umar/ibot

set -e

source /vol/research/fmodel_medical/people/umar/miniconda3/etc/profile.d/conda.sh
conda activate ibot

python -m torch.distributed.launch --nproc_per_node=2 main_ibot.py

