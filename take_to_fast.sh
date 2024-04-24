#!/bin/bash

rsync -av /vol/research/fmodel_medical/people/umar/miniconda3/envs/ibot/ /mnt/fast/nobackup/users/um00109/umar/miniconda3/envs/ibot/

rsync -av --exclude '*.pth' '/vol/research/fmodel_medical/people/umar/ibot/' '/mnt/fast/nobackup/users/um00109/umar/ibot/'

rsync -av '/vol/research/fmodel_medical/people/umar/ibot/iBOT_INet_ViT_S_16_checkpoint.pth' '/mnt/fast/nobackup/users/um00109/umar/ibot/'