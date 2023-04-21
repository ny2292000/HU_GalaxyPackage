cmake_minimum_required(VERSION 3.10)
project(GalaxyProgram)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_BINARY_DIR  ${CMAKE_SOURCE_DIR}/bin)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_COMPILER /usr/lib/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

# Set include directories
include_directories(
        include
        /usr/include/python3.10/
        /usr/lib/python3/dist-packages/numpy/core/include/
        ${NLopt_INCLUDE_DIRS}
)

set(SOURCES src/GalaxyApp.cpp)

add_executable(GalaxyProgram ${SOURCES})

# Set the library as a dependency for the executable
target_link_libraries(GalaxyProgram PRIVATE
        GalaxyLibNonCuda
        stdc++fs
        m
        /usr/lib/x86_64-linux-gnu/libboost_python310.so
        /usr/lib/x86_64-linux-gnu/libpython3.10.so
        NLopt::nlopt
        )

## Add include directories and compile options as needed
add_dependencies( GalaxyProgram GalaxyLibNonCuda)