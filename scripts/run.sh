#! /bin/bash
# Warning, a complete run can take days.
DATASET_DIR="$HOME/suitesparse"
DATASET_FILES_NAMES="../datasets/suitesparse.txt"
BINARY="../build/bin"

SPMV[0]="work_oriented"
SPMV[1]="group_mapped"
SPMV[2]="thread_mapped"
SPMV[3]="original"
SPMV[4]="merge_path"

EXE_PREFIX="loops.spmv"

for i in {0..4} 
do
  echo "kernel,dataset,rows,cols,nnzs,elapsed" >> spmv_log.${SPMV[$i]}.csv 
  while read -r DATA
  do
    echo $BINARY/$EXE_PREFIX.${SPMV[$i]} -m $DATASET_DIR/$DATA
    timeout 60 $BINARY/$EXE_PREFIX.${SPMV[$i]} -m $DATASET_DIR/$DATA >> spmv_log.${SPMV[$i]}.csv 
  done < "$DATASET_FILES_NAMES"
done
