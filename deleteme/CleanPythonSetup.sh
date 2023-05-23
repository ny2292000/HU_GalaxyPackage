#!/bin/bash
python3.9 setup.py clean --all
cd  /project
rm -rf dist
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

