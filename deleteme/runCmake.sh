#!/bin/bash
. ../deletestuff.sh
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
cd ..; rm -rf build; mkdir build; cd build; cmake -DPython_EXECUTABLE=venv38/bin/python  ..
make
cd ..
python setup.py clean --all
python setup.py build_ext --inplace
python setup.py sdist bdist_wheel


