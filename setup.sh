#! /bin/bash

conda create -n rag-ler python=3.10
conda activate rag-ler

pip install -r requirements.txt
pip install flash-attn==2.5.8

# install faiss CPU version, check details at https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
# conda install -c pytorch faiss-cpu=1.8.0

# install faiss GPU+CPU version
conda install -c pytorch -c nvidia faiss-gpu=1.8.0