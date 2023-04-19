cmake_minimum_required(VERSION 3.20)
project(MainProNonCUDA LANGUAGES CXX)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

message("executable path = "  ${EXECUTABLE_OUTPUT_PATH})
message("library path = " ${LIBRARY_OUTPUT_PATH})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(INTER)
set(Boost_INCLUDE_DIR /usr/include/)
set(BOOST_LIBRARY_DIR /usr/lib/x86_64-linux-gnu/)
set(Boost_PYTHON_LIBRARY /usr/lib/x86_64-linux-gnu/libboost_python310.so)
set(BUILD_SHARED_LIBS ON)

# Set the path to the directory containing FindNumPy.cmake
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(NumPy REQUIRED)
find_package(Boost REQUIRED COMPONENTS Python)
find_package(NLopt CONFIG REQUIRED)



# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_COMPILER /usr/lib/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

# Build GalaxyFormationCUDA library
set(PROGRAM_NAME "TestGalaxy")
set(SOURCE_FILES
        src/TestGalaxy.cpp
        src/Galaxy.cpp src/lib0.cpp)

# Set include directories
include_directories(
        ${CMAKE_SOURCE_DIR}/include
        /home/mp74207/anaconda3/include/python3.9
        /home/mp74207/anaconda3/lib/python3.9/site-packages/numpy/core/include
        /usr/include/boost
        /usr/include
)



# Create lib0 library
set(GALAXY_LIB "libGalaxyLibNonCuda")
add_library(${GALAXY_LIB} src/lib0.cpp src/Galaxy.cpp)
target_link_libraries(${GALAXY_LIB}
        stdc++fs
        m
        /home/mp74207/anaconda3/lib/libboost_python.so
        ${Python_LIBRARIES}
        NLopt::nlopt
        )

find_library(GALAXY_LIB libGalaxyLibNonCUDA.so PATHS ${CMAKE_SOURCE_DIR}/lib)

add_executable(${PROGRAM_NAME}  src/TestGalaxy.cpp)
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
