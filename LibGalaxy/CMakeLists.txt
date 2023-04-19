cmake_minimum_required(VERSION 3.20)
project(LibGalaxy LANGUAGES CXX)
# Set the path to the directory containing FindNumPy.cmake

find_package(NLopt CONFIG REQUIRED)
# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_COMPILER /usr/lib/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
message("Lower PYTHON_INCLUDE_DIRS = " ${Python_INCLUDE_DIRS})


# Add include directories
include_directories(
        /usr/include/python3.10
        ${CMAKE_SOURCE_DIR}/include
        ${PYTHON_INCLUDE_DIRS}
        ${NLopt_INCLUDE_DIRS}
)

# Create lib0 library
set(GALAXY_LIB "GalaxyLibNonCuda")
add_library(${GALAXY_LIB} SHARED src/lib0.cpp src/Galaxy.cpp)
target_link_libraries(${GALAXY_LIB}
        stdc++fs
        m
        ${Boost_LIBRARIES}
        ${PYTHON_LIBRARIES}
        NLopt::nlopt
        )


