#!/bin/bash

# List of Python versions
#python_versions=("cp36-cp36m", "cp37-cp37m", "cp38-cp38", "cp39-cp39", "cp310-cp310",  "cp311-cp311" )  # Add more versions here as needed
python_versions=("cp36-cp36m"  "cp37-cp37m")

# List of Python versions


# The path to the Python installations
python_path="/opt/python"

# The directory where the wheels will be saved
output_dir="/project/auditedwheels"

# Iterate over each Python version
for version in "${python_versions[@]}"; do
    echo $version
#    (
# Build the path to the Python binary and pip
    python_bin="$python_path/$version/bin/python"

    # Create a new virtual environment for this iteration
    venv_path="./manylinux_venv_$version"
    if [ -d "$venv_path" ]; then
        rm -rf "$venv_path"
    fi

    if [ -d "HU_Galaxy.egg-info" ]; then
        rm -rf "HU_Galaxy.egg-info"
    fi

    if [ -d "build" ]; then
        rm -rf "build"
    fi

    if [ -d "dist" ]; then
        rm -rf "dist"
    fi

    if [ -d "HU_Galaxy/CMAKEFILES" ]; then
        rm -rf "HU_Galaxy/CMAKEFILES"
    fi
    if [ -d "HU_Galaxy/_deps" ]; then
        rm -rf "HU_Galaxy/_deps"
    fi

    if [ -d "HU_Galaxy.egg-info" ]; then
        rm -rf "HU_Galaxy.egg-info"
    fi

    if [ -f "HU_Galaxy/CMakeCache.txt" ]; then
        rm -rf "HU_Galaxy/CMakeCache.txt"
    fi
    $python_bin -m venv $venv_path
    python_bin_venv="$venv_path/bin/python"
    pip_bin_venv="$venv_path/bin/pip"
    source $venv_path/bin/activate
    # Install necessary packages in the virtual environment
    $pip_bin_venv install numpy pybind11 nlopt wheel

    # Build the extension modules, build the wheel, find the wheel file,
    # use auditwheel to repair the wheel, cleanup the dist directory
    # All these commands are run in a subshell where the virtual environment is activated
        $python_bin_venv setup.py build_ext --inplace
        $python_bin_venv setup.py sdist bdist_wheel
        dist_dir=$(pwd)/dist
        wheel_file=$(find $dist_dir -name "*.whl" -print -quit)
        auditwheel repair $wheel_file -w $output_dir
       if [ -d "build" ]; then
            rm -rf "build"
        fi

        if [ -d "dist" ]; then
            rm -rf "dist"
        fi

        if [ -d "HU_Galaxy/CMAKEFILES" ]; then
            rm -rf "HU_Galaxy/CMAKEFILES"
        fi
        if [ -d "HU_Galaxy/_deps" ]; then
            rm -rf "HU_Galaxy/_deps"
        fi

        if [ -f "HU_Galaxy/CMakeCache.txt" ]; then
            rm -rf "HU_Galaxy/CMakeCache.txt"
        fi



        rm -rf dist
        mkdir dist
        deactivate
#    )

    # Remove the virtual environment
    rm -rf $venv_path
done
