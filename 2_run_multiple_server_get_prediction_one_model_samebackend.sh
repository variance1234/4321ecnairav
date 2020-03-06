#!/bin/bash
RESULT_DIR='/local/user2/Workspace/deeptrainingtest/result_per_epoch'
DATA_DIR='/local/user2/Workspace/crossmodelchecking/data'
done_filename="get_prediction_done.csv"
train_done_filename="train_done.csv"

now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE=$RESULT_DIR"/log/get_prediction_$now.log"

exec &>$LOG_FILE

python 2_run_server_get_prediction_one_model_samebackend.py $DATA_DIR $RESULT_DIR $done_filename $train_done_filename
