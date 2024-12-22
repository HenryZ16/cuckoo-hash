#!/bin/bash

#SBATCH --partition=CS121
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --exclude=sist_gpu[53-56,61-66]

# execute the scripts in "exp" at the root directory of this project cuckoo-hash

# experiment configurations
BUILD_DIR=build
LOG_DIR="log-exp3-2080Ti"
input_file="data.txt"
config_file="config.txt"
repetition=5
num_hash_func=()
size_hash_table=()
cnt_array_key=16777216

for ((i=2; i<=3; i++)); do
  num_hash_func+=( $i )
done
for ((i=1; i<=5; i++)); do
  size_hash_table+=( $(((100+i)*$cnt_array_key/100)) )
done
for ((i=1; i<=10; i++)); do
  size_hash_table+=( $(((10+i)*$cnt_array_key/10)) )
done

echo "the numbers of hash functions are: ${num_hash_func[@]}"
echo "the sizes of hash tables are: ${size_hash_table[@]}"
mkdir $LOG_DIR

# for each configuration
for j in ${num_hash_func[@]}; do
    for k in ${size_hash_table[@]}; do
        echo "Start to perform insert experiment with ${cnt_array_key} keys on the hash table of size ${size_hash_table}, ${j} hash functions for ${repetition} times."
        log_file="${LOG_DIR}/exp3_h${j}_s${k}.log"
        echo "Cuckoo Hash Benchmark Experiment with ${j} hash functions and hash table size ${k}" > $log_file
        echo "----------" >> $log_file
        for ((i=0; i<$repetition; i++)); do
            ${BUILD_DIR}/data_generator $input_file insert $cnt_array_key

            echo "num_hash_func            ${j}" > $config_file
            echo "size_hash_table          ${k}" >> $config_file
            echo "input_file               ${input_file}" >> $config_file
            echo "dump_file                data" >> $config_file
            echo "is_binary                1" >> $config_file
            echo "eviction_chain_increment 64" >> $config_file
            timeout 1m ${BUILD_DIR}/cuckoo_hash $config_file >> $log_file
            echo "----------" >> $log_file
            echo "Repetition ${i} done." >> $log_file
            echo "----------" >> $log_file
            rm -rf data
        done
    done
done