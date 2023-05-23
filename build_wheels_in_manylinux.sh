#!/bin/bash
# This script should be run inside manylinux container

# Array of python versions
pythons=("cp36-cp36m" "cp37-cp37m" "cp38-cp38" "cp39-cp39" "cp310-cp310" "cp311-cp311")
cd /project
# Loop over python versions
for PYBIN in "${pythons[@]}"; do
    # Create a new virtual environment
    /opt/python/${PYBIN}/bin/python -m venv /project/${PYBIN}

    # Activate the virtual environment
    source /project/${PYBIN}/bin/activate
    pip install --upgrade pip

    # Install dependencies in the virtual environment
    pip install numpy pybind11 wheel

    # Set environment variables
    export Python_INCLUDE_DIRS=/opt/python/${PYBIN}/include/$(ls /opt/python/${PYBIN}/include)
    export Python_NumPy_DIRS=$(python -c "import numpy, os; print(numpy.get_include())")

    # Use the python version in the virtual environment to build the wheel
#    python -m pip wheel /project/ -w wheelhouse/
    rm -rf /project/dist
    python /project/setup.py sdist bdist_wheel
    mv /project/dist/* /project/wheelhouse/.

    source deactivate
    rm -rf /project/${PYBIN}
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
