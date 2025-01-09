#!/bin/bash

# experiment configurations
export CUDA_VISIBLE_DEVICES=1
BUILD_DIR=build
LOG_DIR="log-exp1-2080Ti"
input_file="data.txt"
config_file="config.txt"
repetition=5
num_hash_func=()
size_hash_table=33554432
cnt_array_key=()

for ((i=2; i<=3; i++)); do
  num_hash_func+=( $i )
done
for ((i=10; i<=24; i++)); do
  cnt_array_key+=( $((2**i)) )
done

echo "the numbers of hash functions are: ${num_hash_func[@]}"
echo "the numbers of key arrays are: ${cnt_array_key[@]}"
mkdir $LOG_DIR

# for each configuration
for j in ${num_hash_func[@]}; do
    for k in ${cnt_array_key[@]}; do
        log_file="${LOG_DIR}/exp1_h${j}_k${k}.log"
        echo "Dump data from ${log_file}."
        
        values=($(grep -oP 'Time elapsed in inserting \d+ items: \K[0-9]+\.[0-9]+' "${log_file}"))
        count=0
        sum=0

        for value in "${values[@]}"; do
          sum=$(echo "$sum + $value" | bc)
          count=$((count + 1))

          if (( count == 5 )); then
            average=$(echo "scale=6; $sum / $count" | bc)
            echo "Average of group: $average"
            speed=$(echo "scale=6; $k / $average" | bc)
            echo "Speed of group: $speed"
            count=0
            sum=0
          fi
        done

        if (( count > 0 )); then
          average=$(echo "scale=6; $sum / $count" | bc)
          echo "Average of remaining group: $average"
        fi
    done
done


