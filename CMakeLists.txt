cmake_minimum_required(VERSION 3.20)
project(MainProNonCUDA LANGUAGES CXX)
set(CMAKE_BINARY_DIR  ${CMAKE_SOURCE_DIR}/bin)
set(LD_LIBRARY_PATH=/usr/lib/python3.10/:$ENV{LD_LIBRARY_PATH})
# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=4")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -flto=4")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

# Set the path to the directory containing FindNumPy.cmake
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(NumPy REQUIRED)
find_package(NLopt CONFIG REQUIRED)
find_package(pybind11 REQUIRED)



# Add subdirectories
add_subdirectory(LibGalaxyClass)
add_subdirectory(LibGalaxy)
add_subdirectory(GalaxyProgram)
add_subdirectory(HU_Galaxy)