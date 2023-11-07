# /home/ychenfei/research/libs/loops/build/bin/loops.spgemm.thread_mapped -m /home/ychenfei/research/sparse_matrix_perf_analysis/spgemm_dataflow_analysis/test_mtx/s100/ck104.mtx

# filepath="/home/ychenfei/research/sparse_matrix_perf_analysis/spgemm_dataflow_analysis/test_mtx2/bcsstk17.mtx"
# filename=$(basename "$filepath")

# echo "$filename" >> /home/ychenfei/research/libs/loops/examples/spgemm/running_time.txt
# /home/ychenfei/research/libs/loops/build/bin/loops.spgemm.thread_mapped -m /home/ychenfei/research/sparse_matrix_perf_analysis/spgemm_dataflow_analysis/test_mtx2/bcsstk17.mtx >> /home/ychenfei/research/libs/loops/examples/spgemm/running_time.txt

# Output matrix C in dense format
# export_file="/home/ychenfei/research/libs/loops/examples/spgemm/running_time/dense_C/dense_C_running_time_$(date +%Y-%m-%d).txt"

export_file="/home/ychenfei/research/libs/loops/examples/spgemm/running_time/dense_C/testing.txt"

exe="/home/ychenfei/research/libs/loops/build/bin/loops.spgemm.thread_mapped"

> $export_file

for f in /data/toodemuy/datasets/floridaMatrices/*.mtx
do
    filename=$(basename "$f")
    echo "$filename" >> $export_file
    $exe -m $f >> $export_file
    echo >> $export_file
done