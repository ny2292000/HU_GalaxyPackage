cmake_minimum_required(VERSION 3.22.0)
project(HU_GalaxyPackage LANGUAGES CXX)

set(createTest False)
if (createTest)
    message("Creating testLibrary Program")
else()
    message("Creating only the pybind11 library")
endif()


message("this is pybind include dirs " ${pybind11_INCLUDE_DIR} )
add_subdirectory(hugalaxy)

if(createTest)
    add_subdirectory(testCPPLibrary)
    add_dependencies(testCPPLibrary HU_Galaxy_GalaxyWrapper)
endif()



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


#pkg-config --exists nlopt
#pkg-config --libs nlopt
#pkg-config --cflags nlopt
#pkg-config --modversion nlopt
#
#
#pkg-config --exists torch
#pkg-config --libs torch
#pkg-config --cflags torch
#pkg-config --modversion torch
#
#pkg-config --exists pybind11
#pkg-config --libs pybind11
#pkg-config --cflags pybind11
#pkg-config --modversion pybind11