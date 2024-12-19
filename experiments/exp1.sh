#!/bin/bash

#SBATCH --partition=CS121
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --exclude=sist_gpu[53-56,61-66]

# execute the scripts in "exp" at the root directory of this project cuckoo-hash

# experiment configurations
export CUDA_VISIBLE_DEVICES=1
BUILD_DIR=build
LOG_DIR="log-exp1"
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
        echo "Start to perform insert experiment with ${k} keys on the hash table of size ${size_hash_table}, ${j} hash functions for ${repetition} times."
        log_file="${LOG_DIR}/exp1_h${j}_k${k}.log"
        echo "Cuckoo Hash Benchmark Experiment with ${j} hash functions and inserting ${k} keys" > $log_file
        echo "----------" >> $log_file
        for ((i=0; i<$repetition; i++)); do
            ${BUILD_DIR}/data_generator $input_file insert $k

            echo "num_hash_func            ${j}" > $config_file
            echo "size_hash_table          ${size_hash_table}" >> $config_file
            echo "input_file               ${input_file}" >> $config_file
            echo "dump_file                data" >> $config_file
            echo "is_binary                1" >> $config_file
            echo "eviction_chain_increment 64" >> $config_file
            ${BUILD_DIR}/cuckoo_hash $config_file >> $log_file
            echo "----------" >> $log_file
            echo "Repetition ${i} done." >> $log_file
            echo "----------" >> $log_file
            rm -rf data
        done
    done
done