cmake_minimum_required(VERSION 3.20)
project(MainProNonCUDA LANGUAGES CXX)
set(CMAKE_BINARY_DIR  ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

# Set the path to the directory containing FindNumPy.cmake
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(NumPy REQUIRED)
find_package(Boost REQUIRED COMPONENTS Python)
find_package(NLopt CONFIG REQUIRED)



# Add subdirectories
add_subdirectory(LibGalaxy)
add_subdirectory(GalaxyProgram)