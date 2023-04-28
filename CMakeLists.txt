cmake_minimum_required(VERSION 3.20)
project(MainProNonCUDA LANGUAGES CXX)
set(CMAKE_BINARY_DIR  ${CMAKE_SOURCE_DIR}/bin)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=4")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -flto=4")

# Add subdirectories
add_subdirectory(HU_Galaxy)



