#!/bin/bash

LR=${LR:-"0.1"}
BS=${BS:-"256"}
GP_LAMBDA=${GP_LAMBDA:-"1e-6"}

pushd ../
python imagenet_main.py ~/data/imagenet -a resnet50 \
       --dist-url 'tcp://127.0.0.1:12501'\
       --dist-backend 'nccl'\
       --multiprocessing-distributed \
       --world-size 1  \
       --rank 0 \
       --identifier test \
       --decomposition \
       --od-sampler per_layer \
       --no-full-pass \
       --lr $LR \
       --ignore-k-first-layers 43 \
       --ignore-last-layer \
       --full-training-epochs 1 \
       --epochs 2 \
       --gp --gp-lambda ${GP_LAMBDA} \
       --progressive \
       -b $BS
popd