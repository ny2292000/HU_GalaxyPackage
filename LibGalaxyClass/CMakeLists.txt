cmake_minimum_required(VERSION 3.20)
project(GalaxyCPP_Class LANGUAGES CXX)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Add include directories
include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${NLopt_INCLUDE_DIRS}
)

# Create lib0 library
set(SOURCE_FILES src/Galaxy.cpp)
link_directories(${CMAKE_SOURCE_DIR}/lib)
add_library(GalaxyCPP_Class SHARED ${SOURCE_FILES})
set_target_properties(GalaxyCPP_Class PROPERTIES PREFIX ""  OUTPUT_NAME "GalaxyCPP_Class")

target_link_libraries(GalaxyCPP_Class PRIVATE
        GalaxyCPP_Helper
        stdc++fs
        m
        NLopt::nlopt
        )

add_dependencies( GalaxyCPP_Class GalaxyCPP_Helper)
