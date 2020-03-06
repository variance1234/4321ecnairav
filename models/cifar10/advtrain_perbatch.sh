#!/bin/bash

now="$(date +"%y_%m_%d_%H_%M_%S")"
exec &> "train_cifar10_resnet32v1_fgsm_server_$now.log"

source activate /home/user2/.conda/envs/K_2.2.2_tensorflow_1.10.0
CUDA_VISIBLE_DEVICES=3 KERAS_BACKEND="tensorflow" python 'adversarial_training_per_batch.py'
source deactivate
