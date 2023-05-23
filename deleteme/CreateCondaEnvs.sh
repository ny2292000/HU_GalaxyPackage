#!/bin/bash

# List of Python versions
python_versions=("3.6" "3.7" "3.8" "3.9" "3.10" "3.11")


conda="/anaconda3/bin/conda"

# Iterate over each Python version
for version in "${python_versions[@]}"; do
    # Create a new virtual environment for this iteration
    env_name=manylinux_venv_$version
    $conda create -n $env_name python=$version numpy pybind11 nlopt wheel -y
done
