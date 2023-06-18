#!/bin/bash
rm -rf CMakeFiles
rm -rf hugalaxy/detect_cuda_compute_capabilities.cu
rm -rf hugalaxy/detect_cuda_version.cc
rm -rf hugalaxy/cmake_install.cmake
rm -rf hugalaxy/CMakeCache.txt
rm -rf hugalaxy/_deps
rm -rf hugalaxy/hugalaxy
rm -rf hugalaxy/HU_Galaxy_GalaxyWrapper.cpython-*
rm -rf hugalaxy/libHU_Galaxy_GalaxyWrapperLib.so
rm hugalaxy/HU_Galaxy_GalaxyWrapper.cpython-3.9-x86_64-linux-gnu.so
rm -rf hugalaxy/Makefile
rm -rf hugalaxy/CMakeFiles
rm -rf HU_Galaxy.egg-info
rm -rf build
rm -rf dist
rm -rf cmake-build-release
rm -rf manylinux_venv_c*
rm -rf auditedwheels/*
rm -f HU_Galaxy.cpython*
rm -f cmake-build-debug/CMakeCache.txt
rm -rf wheelhouse/*
rm /home/mp74207/CLionProjects/HU_GalaxyPackage/hugalaxy/HU_Galaxy_GalaxyWrapper.cpython-39-x86_64-linux-gnu.so
pip uninstall -y hugalaxy
cd hugalaxy; cmake . ; make; cd ..
rm -rf hugalaxy/CMakeFiles
rm -rf hugalaxy/CMakeCache.txt
rm hugalaxy/Makefile
rm hugalaxy/cmake_install.cmake
rm -rf hugalaxy/detect_cuda_compute_capabilities.cu
rm -rf hugalaxy/detect_cuda_version.cc
python -m build .
pip install dist/hugalaxy-0.0.1-py3-none-any.whl
deactivate
conda activate Cosmos
pip uninstall -y hugalaxy
pip install dist/hugalaxy-0.0.1-py3-none-any.whl
python -c "import hugalaxy.HU_Galaxy_GalaxyWrapper as hh; print(hh.__dir__())"


#pip install dist/hugalaxy-0.1-cp39-cp39-linux_x86_64.whl
#python -c "from hugalaxy import GalaxyWrapper as GW"