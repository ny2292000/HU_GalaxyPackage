#!/bin/bash
export Python_EXECUTABLE=`which python`
export Python_INCLUDE_DIRS=`python -c "import sysconfig; print(sysconfig.get_paths()['include'])"`
export NUMPY_INCLUDE_DIRS=$(python -c "import numpy, os; print(numpy.get_include())")
#export pybind11_INCLUDE_DIR=$(python -c "import pybind11, os; print(pybind11.get_include())")
export Pytorch_LIB_DIR="/usr/lib/x86_64-linux-gnu"
export TORCH_LIBRARY="/usr/lib/x86_64-linux-gnu/libtorch.so.1.8.0"
export CMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
printenv |grep -i python
#printenv |grep -i pybind11
