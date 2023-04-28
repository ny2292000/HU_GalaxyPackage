cmake_minimum_required(VERSION 3.12)
project(HU_Galaxy_GalaxyWrapper)

list( APPEND LD_LIBRARY_PATH "/usr/local/lib/python3.10/")
list(APPEND CMAKE_PREFIX_PATH "/usr/local")


find_package(pybind11 REQUIRED)
# Find Python and related packages
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
# Find other required packages
find_package(NLopt CONFIG REQUIRED)


#set(PYTHON_EXECUTABLE /usr/local/bin/python3.10)

# Set C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)

# Add include directories
include_directories(
        /usr/local/include
        ${pybind11_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PYTHON_INCLUDE_DIRS}
        ${NLopt_INCLUDE_DIRS}
        /usr/lib/python3/dist-packages/numpy/core/include
        /usr/include/
)





# Add source files
pybind11_add_module(${PROJECT_NAME}
#add_library(${PROJECT_NAME} SHARED
        HU_Galaxy.cpp
        Galaxy.cpp
        )

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
        ${PYTHON_LIBRARIES}
        ${NUMPY_LIBRARIES}
        pybind11::module
        NLopt::nlopt
        m
        stdc++fs
        )

