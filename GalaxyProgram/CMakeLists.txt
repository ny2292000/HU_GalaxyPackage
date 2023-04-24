cmake_minimum_required(VERSION 3.10)
project(GalaxyProgram)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_BINARY_DIR  ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_PATH ${CMAKE_SOURCE_DIR}/lib)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_COMPILER /usr/lib/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)


#set(CMAKE_VERBOSE_MAKEFILE ON)
message("Compiling " ${CMAKE_PROJECT_NAME})

# Find pybind11 package
find_package(pybind11 CONFIG REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(NumPy REQUIRED)
find_package(NLopt CONFIG REQUIRED)


# Set include directories
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        /usr/include
        /usr/local/include
        /usr/include/python3.10
        ${NLopt_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIRS}
        /usr/lib/python3/dist-packages/numpy/core/include
)

#set(SOURCES src/GalaxyApp.cpp src/Galaxy.cpp include/Galaxy.h)
set(SOURCES src/GalaxyApp.cpp  include/Galaxy.h)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_executable(GalaxyProgram ${SOURCES})

# Set the library as a dependency for the executable
target_link_libraries(GalaxyProgram PUBLIC
        HU_Galaxy_GalaxyWrapper
        ${PYTHON_LIBRARIES}
        ${NUMPY_LIBRARIES}
        stdc++fs
        m
        NLopt::nlopt
        pybind11::lto
        pybind11::embed
        pybind11::module
        pybind11::headers
        )

## Add include directories and compile options as needed
#add_dependencies( GalaxyProgram libGalaxyClassCPP)
