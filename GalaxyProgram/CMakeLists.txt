cmake_minimum_required(VERSION 3.10)
project(GalaxyProgram)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_BINARY_DIR  ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_PATH ${CMAKE_SOURCE_DIR}/lib)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_COMPILER /usr/lib/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
# Set include directories
include_directories(
        include
        ${NLopt_INCLUDE_DIRS}
)

set(SOURCES src/GalaxyApp.cpp)

add_executable(GalaxyProgram ${SOURCES})

# Set the library as a dependency for the executable
target_link_libraries(GalaxyProgram PRIVATE
        ${CMAKE_LIBRARY_PATH}/GalaxyLibNonCuda.so
        stdc++fs
        m
        NLopt::nlopt
        )

## Add include directories and compile options as needed
add_dependencies( GalaxyProgram GalaxyLibNonCuda)