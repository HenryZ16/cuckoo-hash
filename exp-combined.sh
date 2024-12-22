#!/bin/bash

#SBATCH --partition=CS121
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --exclude=sist_gpu[53-56,61-66]

export LD_LIBRARY_PATH=/public/software/gcc/gcc-11.2.0/lib64/:/public/software/CUDA/cuda-12.4/lib64/:$LD_LIBRARY_PATH

./experiments/exp1.sh
./experiments/exp2.sh
./experiments/exp3.sh
./experiments/exp4.sh