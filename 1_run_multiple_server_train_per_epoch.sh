#!/bin/bash
RESULT_DIR='/local/user2/Workspace/deeptrainingtest/result_per_epoch'
#RESULT_DIR='/local/user2/Workspace/deeptrainingtest/result_per_epoch_deterministic'
DATA_DIR='/local/user2/Workspace/crossmodelchecking/data'
done_filename="train_done.csv"

now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE=$RESULT_DIR"/log/train_per_epoch_$now.log"

exec &>$LOG_FILE

python 1_run_server_train_per_epoch.py $DATA_DIR $RESULT_DIR $done_filename
