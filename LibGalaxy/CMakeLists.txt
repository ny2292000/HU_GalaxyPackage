cmake_minimum_required(VERSION 3.20)
project(LibGalaxy LANGUAGES CXX)
# Set the path to the directory containing FindNumPy.cmake

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Add include directories
include_directories(
        /usr/include/python3.10/
        /usr/lib/python3/dist-packages/numpy/core/include/
        ${CMAKE_SOURCE_DIR}/include
        ${PYTHON_INCLUDE_DIRS}
        ${NLopt_INCLUDE_DIRS}
)

# Create lib0 library
set(GALAXY_LIB "GalaxyLibNonCuda")
add_library(${GALAXY_LIB} SHARED src/lib0.cpp src/Galaxy.cpp)

target_link_libraries(${GALAXY_LIB} PRIVATE
        stdc++fs
        m
        /usr/lib/x86_64-linux-gnu/libboost_python310.so
        /usr/lib/x86_64-linux-gnu/libpython3.10.so
        NLopt::nlopt
        )


