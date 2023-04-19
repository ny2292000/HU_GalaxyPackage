cmake_minimum_required(VERSION 3.20)
project(GalaxyProgram LANGUAGES CXX)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_COMPILER /usr/lib/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

# Build GalaxyFormationCUDA library
set(PROGRAM_NAME "TestGalaxy")
set(SOURCE_FILES
        src/GalaxyApp.cpp
        src/Galaxy.cpp src/lib0.cpp)

# Set include directories
include_directories(
        ${PROJECT_SOURCE_DIR}/include
        /home/mp74207/anaconda3/include/python3.9
        /home/mp74207/anaconda3/lib/python3.9/site-packages/numpy/core/include
        /usr/include/boost
        /usr/include
)



find_library(GALAXY_LIB libGalaxyLibNonCUDA.so PATHS ${CMAKE_SOURCE_DIR}/lib)

add_executable(${PROGRAM_NAME} src/GalaxyApp.cpp)
target_link_libraries(${PROGRAM_NAME} PRIVATE
        ${GALAXY_LIB}
        /home/mp74207/anaconda3/lib/libpython3.9.so
        /home/mp74207/anaconda3/lib/libboost_python.so
        ${PYTHON_LIBRARIES}
        ${NUMPY_LIBRARIES}
        stdc++fs
        m
        NLopt::nlopt
        )


add_test(
        NAME test1
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/${PROGRAM_NAME}
)

set_property(
        TEST test1
        PROPERTY ENVIRONMENT LD_LIBRARY_PATH=${LIBRARY_OUTPUT_PATH}
)
