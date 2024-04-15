#!/bin/bash

cd /vol/research/fmodel_medical/people/umar/ibot
conda activate ibot

pip3 install -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html mmcv-full==1.3.9
pip3 install pytest-runner scipy tensorboardX faiss-gpu==1.6.1 tqdm lmdb sklearn pyarrow==2.0.0 timm DALL-E munkres six einops

# install apex
pip3 install git+https://github.com/NVIDIA/apex --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"