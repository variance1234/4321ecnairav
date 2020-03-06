#!/bin/bash
RESULT_DIR='/local/user2/Workspace/deeptrainingtest/result_per_epoch'
DATA_DIR='/local/user2/Workspace/crossmodelchecking/data'

now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE=$RESULT_DIR"/log/analyze_$now.log"

exec &>$LOG_FILE

source activate K_2.2.2_tensorflow_1.10.0
python 3_0_analyze_network_outputs.py $DATA_DIR $RESULT_DIR
source deactivate
