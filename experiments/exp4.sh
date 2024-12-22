#!/bin/bash

#SBATCH --partition=CS121
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --exclude=sist_gpu[53-56,61-66]

# execute the scripts in "exp" at the root directory of this project cuckoo-hash

# experiment configurations
BUILD_DIR=build
LOG_DIR="log-exp4-2080Ti"
input_file="data.txt"
config_file="config.txt"
repetition=5
num_hash_func=()
size_hash_table=23488102
cnt_array_key=16777216
eviction_chain_increment=()

for ((i=2; i<=3; i++)); do
  num_hash_func+=( $i )
done
for ((i=0; i<=8; i++)); do
  eviction_chain_increment+=( $((2**i)) )
done

echo "the numbers of hash functions are: ${num_hash_func[@]}"
echo "the numbers of key arrays are: ${cnt_array_key[@]}"
mkdir $LOG_DIR

# for each configuration
for j in ${num_hash_func[@]}; do
    k=$cnt_array_key
      for l in ${eviction_chain_increment[@]}; do
        echo "Start to perform insert experiment with ${k} keys on the hash table of size ${size_hash_table}, ${j} hash functions for ${repetition} times. The eviction chain increment is ${l}"
        log_file="${LOG_DIR}/exp1_h${j}_e${l}.log"
        echo "Cuckoo Hash Benchmark Experiment with ${j} hash functions and inserting ${k} keys" > $log_file
        echo "----------" >> $log_file
        for ((i=0; i<$repetition; i++)); do
            ${BUILD_DIR}/data_generator $input_file insert $k

            echo "num_hash_func            ${j}" > $config_file
            echo "size_hash_table          ${size_hash_table}" >> $config_file
            echo "input_file               ${input_file}" >> $config_file
            echo "dump_file                data" >> $config_file
            echo "is_binary                1" >> $config_file
            echo "eviction_chain_increment ${l}" >> $config_file
            timeout 1m ${BUILD_DIR}/cuckoo_hash $config_file >> $log_file
            echo "----------" >> $log_file
            echo "Repetition ${i} done." >> $log_file
            echo "----------" >> $log_file
            rm -rf data
        done
      done
done