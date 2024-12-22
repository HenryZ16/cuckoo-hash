#!/bin/bash
BUILD_DIR=build
N_PROC=$(lscpu | grep "^CPU(s):" | awk '{print $2}')

GCC_PATH=/public/software/gcc/gcc-11.2.0/bin/g++

echo "[build.sh] Building the project"
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake -DCMAKE_CXX_COMPILER=${GCC_PATH} \
      -DDEBUG=false \
      -DBENCHMARK=true \
      -S ..
mv compile_commands.json ..
make -j ${N_PROC}
