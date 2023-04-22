cmake_minimum_required(VERSION 3.20)
project(GalaxyCPP_Helper LANGUAGES CXX)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Add include directories
include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${NLopt_INCLUDE_DIRS}
)

# Create lib0 library
set(SOURCE_FILES src/lib0.cpp)
add_library(GalaxyCPP_Helper SHARED ${SOURCE_FILES})
set_target_properties(GalaxyCPP_Helper PROPERTIES PREFIX ""  OUTPUT_NAME "GalaxyCPP_Helper")

target_link_libraries(GalaxyCPP_Helper PRIVATE
        stdc++fs
        m
        NLopt::nlopt
        )
#add_dependencies(GalaxyCPP_Helper GalaxyCPP_Class)