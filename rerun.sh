#!/bin/bash


rm -rf CMakeFiles
rm -rf HU_Galaxy/cmake_install.cmake
rm -rf HU_Galaxy/CMakeCache.txt
rm -rf HU_Galaxy/_deps
rm -rf HU_Galaxy/HU_Galaxy_GalaxyWrapper.cpython-*
rm -rf HU_Galaxy/libHU_Galaxy_GalaxyWrapperLib.so
rm -rf HU_Galaxy/Makefile
rm -rf HU_Galaxy/CMakeFiles
rm -rf HU_Galaxy.egg-info
rm -rf build
rm -rf dist
rm -rf cmake-build-release
rm -rf manylinux_venv_c*
rm -rf auditedwheels/*
rm -f HU_Galaxy.cpython*
rm -f cmake-build-debug/CMakeCache.txt
rm -rf wheelhouse/*
rm -rf `find ./venv/lib/python3.10/site-packages/  |grep HU_Galaxy`
python setup.py sdist bdist_wheel
pip install dist/HU_Galaxy-0.2.tar.gz
python -c "import HU_Galaxy_GalaxyWrapper as HUG;print(HUG.__dir__());"