#!/bin/bash
BUILD_DIR=build
N_PROC=$(lscpu | grep "^CPU(s):" | awk '{print $2}')

GCC_PATH=$(spack find --format="{prefix}/bin/g++" gcc@13.2)

echo "[build.sh] Building the project"
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake -DCMAKE_CXX_COMPILER=${GCC_PATH} \
      -S ..
mv compile_commands.json ..
make -j ${N_PROC}