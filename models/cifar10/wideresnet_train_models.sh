#!/bin/bash

now="$(date +"%y_%m_%d_%H_%M_%S")"
exec &> "train_cifar10_widresnet_34_10_server_$now.log"

source activate K_2.2.2_tensorflow_1.10.0
CUDA_VISIBLE_DEVICES=0 KERAS_BACKEND="tensorflow" python 'wideresnet_training.py'
source deactivate
