cmake_minimum_required(VERSION 3.20)
project(galaxy_cpp_helper VERSION 1.0)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
# Set C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MESSAGE_LOG_LEVEL "VERBOSE")

#
#set(CMAKE_VERBOSE_MAKEFILE ON)
message("Compiling " ${CMAKE_PROJECT_NAME})

# Find pybind11 package
find_package(pybind11 CONFIG REQUIRED)
message("pybind11 found:"  ${pybind11_FOUND})
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
message("Python found:"  ${Python_FOUND})
find_package(NumPy REQUIRED)
message("NUMPY found: ${NumPy_FOUND}")
find_package(NLopt CONFIG REQUIRED)
message("NLopt found:"  ${NLopt_FOUND})


# Add include directories
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${pybind11_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIRS}
        ${NLOPT_INCLUDE_DIRS}
)

# Add source files
add_library(${PROJECT_NAME} SHARED
        src/lib0.cpp
        )

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
        pybind11::module
        ${PYTHON_LIBRARIES}
        ${NUMPY_LIBRARIES}
        ${NLOPT_LIBRARIES}
        PUBLIC
        nlopt
        stdc++fs
        m
        /usr/lib/x86_64-linux-gnu/libnlopt_cxx.so
        NLopt::nlopt
        pybind11::lto
        pybind11::embed
        pybind11::module
        pybind11::headers
        )


# Set module dependencies
set(MODULE_DEPENDENCIES "")
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "${MODULE_DEPENDENCIES}")
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,${MODULE_DEPENDENCIES}")

# Set output module name and location
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "${PROJECT_NAME}" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib")
