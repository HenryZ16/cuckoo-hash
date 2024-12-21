#!/bin/bash

# experiment configurations
export CUDA_VISIBLE_DEVICES=1
BUILD_DIR=build
LOG_DIR="log-exp2"
input_file="data"
config_file="config.txt"
repetition=5
num_hash_func=()
size_hash_table=33554432
cnt_array_key=16777216

for ((i=2; i<=3; i++)); do
  num_hash_func+=( $i )
done

echo "the numbers of hash functions are: ${num_hash_func[@]}"
echo "the numbers of key arrays are: ${cnt_array_key[@]}"
mkdir $LOG_DIR

# for each configuration
for j in ${num_hash_func[@]}; do
    for k in ${cnt_array_key[@]}; do
        log_file="${LOG_DIR}/exp2_h${j}_k${k}.log"
        echo "Dump data from ${log_file}."

        # Dump insert
        values=($(grep -oP 'Time elapsed in inserting \d+ items: \K[0-9]+\.[0-9]+' "${log_file}"))
        count=0
        sum=0

        echo "Group data_insert"
        for value in "${values[@]}"; do
          sum=$(echo "$sum + $value" | bc)
          count=$((count + 1))

          if (( count == 5 )); then
            average=$(echo "scale=2; $sum / $count" | bc)
            echo "Average of group: $average"
            speed=$(echo "scale=2; $k / $average" | bc)
            echo "Speed of group: $speed"
            count=0
            sum=0
          fi
        done

        if (( count > 0 )); then
          average=$(echo "scale=2; $sum / $count" | bc)
          echo "Average of remaining group: $average"
        fi

        # Dump lookup
        for((m=0; m<=10; m++)); do
          values=()
          input_file="data_lookup_${m}"
          line_number=$(grep -n "^input_file: ${input_file}$" "${log_file}" | cut -d: -f1)
          if [ -n "$line_number" ]; then
            for line in $line_number; do
              target_line=$((line + 7))
              values+=( $(sed -n "${target_line}p" "${log_file}" | grep -oP 'Time elapsed in looking up \d+ items: \K[0-9]+\.[0-9]+') )
            done
          fi

          count=0
          sum=0

          echo "Group data_lookup_${m}"
          for value in "${values[@]}"; do
            sum=$(echo "$sum + $value" | bc)
            count=$((count + 1))

            if (( count == 5 )); then
              average=$(echo "scale=2; $sum / $count" | bc)
              echo "Average of group: $average"
              speed=$(echo "scale=2; $k / $average" | bc)
              echo "Speed of group: $speed"
              count=0
              sum=0
            fi
          done

          if (( count > 0 )); then
            average=$(echo "scale=2; $sum / $count" | bc)
            echo "Average of remaining group: $average"
          fi
        done
    done
done


