#!/bin/bash

now="$(date +"%y_%m_%d_%H_%M_%S")"
exec &> "train_cifar10_resnet_32_server_$now.log"

source activate K_2.2.2_tensorflow_1.10.0
CUDA_VISIBLE_DEVICES=1 KERAS_BACKEND="tensorflow" python 'resnet_training.py'
source deactivate
