#!/bin/bash

#SBATCH --partition=CS121
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --exclude=sist_gpu[53-56,61-66]

export OMP_NUM_THREADS=12
export LD_LIBRARY_PATH=/public/software/gcc/gcc-11.2.0/lib64/:/public/software/CUDA/cuda-12.4/lib64/:$LD_LIBRARY_PATH
BUILD_DIR=build

${BUILD_DIR}/cuckoo_hash config.txt

rm -rf data