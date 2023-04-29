#!/bin/bash

## Find shared libraries containing the word "python"
#find /usr/local -iname "*python*.so*" -type f | while read -r library; do
#  echo "Checking library: ${library}"
#  nm_output=$(nm "${library}" 2>/dev/null | grep -i 'PyThreadState_Get')
#
#  if [[ ! -z "${nm_output}" ]]; then
#    echo "Symbol found in library: ${library}"
#    echo "${nm_output}"
#  else
#    echo "Symbol not found in library: ${library}"
#  fi
#
#  echo "-------------------------"
#done

python setup.py clean --all
rm  -rf /home/mp74207/CLionProjects/GalaxyFormationSingleFile/dist
rm -rf ./build
rm -rf ./HU_Galaxy.egg-info
rm ./HU_Galaxy/cmake_install.cmake
rm ./HU_Galaxy/CMakeCache.txt
rm -rf ./HU_Galaxy/CMakeFiles
rm ./HU_Galaxy/HU_Galaxy_GalaxyWrapper.cpython-39-x86_64-linux-gnu.so
rm ./HU_Galaxy/Makefile
rm ./HU_Galaxy.cpython-39-x86_64-linux-gnu.so
rm HU_Galaxy.cpython-39-x86_64-linux-gnu.so
python setup.py clean --all
pip uninstall HU_Galaxy -y
python setup.py build_ext --inplace
python setup.py install

python setup.py clean --all
rm  -rf /home/mp74207/CLionProjects/GalaxyFormationSingleFile/dist
rm -rf ./build
rm -rf ./HU_Galaxy.egg-info
rm ./HU_Galaxy/cmake_install.cmake
rm ./HU_Galaxy/CMakeCache.txt
rm -rf ./HU_Galaxy/CMakeFiles
rm ./HU_Galaxy/HU_Galaxy_GalaxyWrapper.cpython-39-x86_64-linux-gnu.so
rm ./HU_Galaxy/Makefile
rm ./HU_Galaxy.cpython-39-x86_64-linux-gnu.so
rm HU_Galaxy.cpython-39-x86_64-linux-gnu.so
python setup.py clean --all