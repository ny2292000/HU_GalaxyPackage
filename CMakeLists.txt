cmake_minimum_required(VERSION 3.22.0)
project(HU_GalaxyPackage LANGUAGES CXX)
set(CMAKE_STRIP "")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 50 86)

# Set the virtual environment path
set(VENV_PATH "${CMAKE_SOURCE_DIR}/myvenv")
message(" this is the path to venv "  ${VENV_PATH})
# Set the necessary environment variables
set(ENV{VIRTUAL_ENV} ${VENV_PATH})
set(ENV{PATH} "${VENV_PATH}/bin:$ENV{PATH}")

set(Python_INCLUDE_DIRS "/usr/include/python3.10")
set(CMAKE_PREFIX_PATH "/home/mp74207/CLionProjects/HU_GalaxyPackage/myvenv/lib/python3.10/site-packages/torch/share/cmake")
set(NUMPY_INCLUDE_DIRS "/home/mp74207/CLionProjects/HU_GalaxyPackage/myvenv/lib/python3.10/site-packages/numpy/core/include")
set(pybind11_INCLUDE_DIR "/home/mp74207/CLionProjects/HU_GalaxyPackage/pybind11/include")
set(CONDA_PYTHON_EXE "/home/mp74207/anaconda3/bin/python")
set(Python_EXECUTABLE "/home/mp74207/CLionProjects/HU_GalaxyPackage/myvenv/bin/python")

set(CUDA_NVRTC_LIB "/home/mp74207/CLionProjects/HU_GalaxyPackage/myvenv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.11.2")
set(CMAKE_CUDA_COMPILER "/usr/bin/nvcc")


find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
message("this is python include dirs " ${Python_INCLUDE_DIRS} )
message("this is numpy include dirs " ${Python_NumPy_INCLUDE_DIRS} )

#set(pybind11_DIR "/home/mp74207/CLionProjects/HU_GalaxyPackage/pybind11/tools")
find_package(pybind11 CONFIG REQUIRED)
message("this is pybind include dirs " ${pybind11_INCLUDE_DIRS} )

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=4 -lnlopt ${TORCH_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -flto=4 -lnlopt ${TORCH_C_FLAGS}")
message("Torch includes" "${TORCH_INCLUDE_DIRS}")
message("Torch libraries" "${TORCH_LIBRARIES}")

#/usr/share/cmake-3.22/Modules/FindPython/Support.cmake:3330
set( CMAKE_MODULE_PATH  ${CMAKE_SOURCE_DIR}/cmake)
include("/home/mp74207/CLionProjects/HU_GalaxyPackage/cmake/FindNLopt.cmake")
find_package(NLopt CONFIG REQUIRED)
if (${PC_NLOPT_FOUND})
    message("found nlopt "  )
else()
    message("not found nlopt "  )
endif()
message("this is nlopt library dirs " ${NLOPT_LIBRARIES} )




message("this is pybind include dirs " ${pybind11_INCLUDE_DIRS} )
add_subdirectory(HU_Galaxy)
add_subdirectory(testCPPLibrary)

add_dependencies(testCPPLibrary HU_Galaxy_GalaxyWrapper_A)


#conda create -n pytorch_env
#conda install pytorch torchvision torchaudio pytorch-cuda=11.8 nlopt pybind11  -c pytorch -c nvidia
#conda install -c anaconda _ipyw_jlab_nb_ext_conf
#conda install -c conda-forge nb_conda nb_conda_kernels

#export Python_EXECUTABLE=`which python`
#export Python_INCLUDE_DIRS=`python -c "import sysconfig; print(sysconfig.get_paths()['include'])"`
#export NUMPY_INCLUDE_DIRS=$(python -c "import numpy, os; print(numpy.get_include())")
#export pybind11_INCLUDE_DIR=$(python -c "import pybind11, os; print(pybind11.get_include())")
#export CMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
#printenv |grep -i python
#printenv |grep -i pybind11