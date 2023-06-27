#!/bin/bash
# This script should be run inside manylinux container
 bash ./createActivateScripts.sh
# Array of python versions
pythons=("cp36-cp36m" "cp37-cp37m" "cp38-cp38" "cp39-cp39" "cp310-cp310" "cp311-cp311")
cd /project
PYBIN="cp38-cp38"
# Loop over python versions
for PYBIN in "${pythons[@]}"; do
    # Create a new virtual environment
    echo ${PYBIN}

    # Install dependencies in the specific Python version
    PYTHON_PATH_ACTIVATEME="/opt/python/${PYBIN}/bin/activate"
    source $PYTHON_PATH_ACTIVATEME
    PIP_PATH="/opt/python/${PYBIN}/bin/pip"
    PYTHON_PATH="/opt/python/${PYBIN}/bin/python"
    $PIP_PATH install numpy pybind11 nlopt wheel cython

    export CFLAGS="$CFLAGS -I$($PYTHON_PATH -c 'import pybind11; print(pybind11.get_include())')"
    export CFLAGS="$CFLAGS -I$($PYTHON_PATH -c 'import numpy; print(numpy.get_include())')"
    export CXXFLAGS="$CXXFLAGS -I$($PYTHON_PATH -c 'import pybind11; print(pybind11.get_include())')"
    export CXXFLAGS="$CXXFLAGS -I$($PYTHON_PATH -c 'import numpy; print(numpy.get_include())')"
    export CFLAGS="$CFLAGS -I$($PYTHON_PATH -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
    export CXXFLAGS="$CXXFLAGS -I$($PYTHON_PATH -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
    export CFLAGS="$CFLAGS -fPIC"
    export CXXFLAGS="$CXXFLAGS -fPIC"
    export Python_INCLUDE_DIRS=$($PYTHON_PATH -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
    export Python_NumPy_DIRS=$($PYTHON_PATH -c "import numpy, os; print(numpy.get_include())")

    export PATH=/opt/python/${PYBIN}/bin:$PATH
    export Python_EXECUTABLE=$PYTHON_PATH

    # Use the python version in the virtual environment to build the wheel
    rm -rf /project/dist
    $PYTHON_PATH /project/setup.py sdist bdist_wheel
    mv /project/dist/* /project/wheelhouse/.

    # Cleanup
    rm -rf /project/CMakeFiles
    rm -rf /project/HU_Galaxy/cmake_install.cmake
    rm -rf /project/HU_Galaxy/CMakeCache.txt
    rm -rf /project/HU_Galaxy/_deps
    rm -rf /project/HU_Galaxy/HU_Galaxy_GalaxyWrapper.cpython-*
    rm -rf /project/HU_Galaxy/Makefile
    rm -rf /project/HU_Galaxy/CMakeFiles
    rm -rf /project/HU_Galaxy.egg-info
    rm -rf /project/build
    rm -rf /project/dist
    rm -rf /project/cmake-build-release
    rm -f /project/HU_Galaxy.cpython*
    rm -f /project/cmake-build-debug/CMakeCache.txt
done

# Now repair the wheels using auditwheel
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" -w /project/wheelhouse/
done

