#!/bin/bash

gen_fast_range() {
    local end=$1
    local i=0
    
    while [ $i -lt "$end" ]; do    
        echo $i
        i=$(( $(printf "%.0f" $(echo "$i * 1.05 + 1" | bc -l)) ))
    done
    echo "$end"
}

TOTAL_ACTIVITIES=9068
TEST_SIZE=1000

for i in $(gen_fast_range $(($TOTAL_ACTIVITIES - $TEST_SIZE))); do
	sbatch run_ltr_context_eval.sh --size "$i"
done