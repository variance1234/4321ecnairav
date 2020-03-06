#!/bin/bash
RESULT_DIR='/local/user2/Workspace/deeptrainingtest/result_inference'
DATA_DIR='/local/user2/Workspace/crossmodelchecking/data'
done_filename="inference_done.csv"

now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE=$RESULT_DIR"/log/inference_$now.log"

exec &>$LOG_FILE

python 3_run_server_inference_one_model.py $DATA_DIR $RESULT_DIR $done_filename
