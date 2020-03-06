#!/bin/bash

ENV=${1}
COMMAND=${2}
LOG_FILE=${3}

source /usr/local/anaconda3/etc/profile.d/conda.sh

export LD_LIBRARY_PATH=/usr/local/mklml/mklml_lnx_2018.0.3.20180406/lib:/usr/local/anaconda3/lib:$LD_LIBRARY_PATH

cd /working

exec &>$LOG_FILE

conda activate $ENV
eval $COMMAND
conda deactivate
