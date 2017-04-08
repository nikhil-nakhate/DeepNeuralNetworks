#!/bin/bash

export TRAIN_PATH=~/data/images/trainset/
export TEST_PATH=~/data/images/testset/
export TUNE_PATH=~/data/images/tuneset/
export IMG_SIZE=32
# rm *.class


make clean;
make


echo "Compilation 	done";
# java Lab2 $DATA_FILE
java Lab3 ${TRAIN_PATH} ${TUNE_PATH} ${TEST_PATH} ${IMG_SIZE}
