#!/bin/bash

# Change to the directory containing the libraries
cd /home/mp74207/CLionProjects/GalaxyFormationSingleFile/lib

# Loop through all files in the directory
for file in *; do
  # Check if the file has a .so extension and doesn't start with 'lib'
  if [[ ${file: -3} == ".so" && ${file:0:3} != "lib" ]]; then
    # Add 'lib' prefix to the file name
    mv "$file" "lib$file"
  fi
done
