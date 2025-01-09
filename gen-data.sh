#!/bin/bash
BUILD_DIR=build
# 33554432
# 16777216

export LD_LIBRARY_PATH=/public/software/gcc/gcc-11.2.0/lib64/:/public/software/CUDA/cuda-12.4/lib64/:$LD_LIBRARY_PATH

${BUILD_DIR}/data_generator data.txt insert 16777216