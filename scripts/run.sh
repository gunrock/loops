#! /bin/bash
# Warning, a complete run can take days.
DATASET_DIR="/data/suitesparse_dataset"
DATASET_FILES_NAMES="../datasets/suitesparse.txt"
BINARY="../build/bin"
ALG=${1:-"4"}

SPMV[0]="work_oriented"
SPMV[1]="group_mapped"
SPMV[2]="thread_mapped"
SPMV[3]="original"
SPMV[4]="merge_path"

EXE_PREFIX="loops.spmv"

for i in {0..4} 
do
  while read -r DATA
  do
    echo $BINARY/$EXE_PREFIX.${SPMV[$ALG]} -m $DATASET_DIR/$DATA
    timeout 200 $BINARY/$EXE_PREFIX.${SPMV[$ALG]} -m $DATASET_DIR/$DATA >> spmv_log.${SPMV[$ALG]}.txt 
  done < "$DATASET_DIR/$DATASET_FILES"
done