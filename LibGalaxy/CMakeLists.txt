cmake_minimum_required(VERSION 3.20)
project(GalaxyLibNonCuda LANGUAGES CXX)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Add include directories
include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${NLopt_INCLUDE_DIRS}
)

# Create lib0 library
set(SOURCE_FILES src/Galaxy.cpp src/lib0.cpp)
add_library(GalaxyLibNonCuda SHARED ${SOURCE_FILES})
set_target_properties(GalaxyLibNonCuda PROPERTIES PREFIX ""  OUTPUT_NAME "GalaxyLibNonCuda")

target_link_libraries(GalaxyLibNonCuda PRIVATE
        stdc++fs
        m
        NLopt::nlopt
        )


