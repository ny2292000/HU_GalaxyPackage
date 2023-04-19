cmake_minimum_required(VERSION 3.20)
project(MainProNonCUDA LANGUAGES CXX)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

# Set the path to the Anaconda Python environment
set(ANACONDA_ENV $ENV{HOME}/anaconda3/envs/Cosmos)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Add subdirectories
add_subdirectory(LibGalaxy)


# Set the environment for module1 tests
#set_property(
#        TEST LibGalaxy_test
#        PROPERTY ENVIRONMENT
#        "PATH=${ANACONDA_ENV}/bin:$ENV{PATH}"
#        "LD_LIBRARY_PATH=${ANACONDA_ENV}/lib:$ENV{LD_LIBRARY_PATH}"
#)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
#set(Boost_INCLUDE_DIR /usr/include/)
#set(BOOST_LIBRARY_DIR /usr/lib/x86_64-linux-gnu/)
#set(Boost_PYTHON_LIBRARY /usr/lib/x86_64-linux-gnu/libboost_python310.so)
#set(BUILD_SHARED_LIBS ON)

# Set the path to the directory containing FindNumPy.cmake
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(NumPy REQUIRED)
find_package(Boost REQUIRED COMPONENTS Python)
find_package(NLopt CONFIG REQUIRED)

message("TOP Anaconda = " ${ANACONDA_ENV})
message("TOP PYTHON_INCLUDE_DIRS = " ${Python_INCLUDE_DIRS})

#include_directories(${Python_INCLUDE_DIRS})
# Set Boost and NLopt include directories
#set(Boost_INCLUDE_DIRS "/usr/include")
set(NLopt_INCLUDE_DIRS "/usr/local/include/nlopt")
#set(Python_INCLUDE_DIRS ${ANACONDA_ENV}/include)



# Export these variables to the build environment
#set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS} CACHE INTERNAL "")
set(NLopt_INCLUDE_DIRS ${NLopt_INCLUDE_DIRS} CACHE INTERNAL "")
#set(Python_INCLUDE_DIRS ${Python_INCLUDE_DIRS} CACHE INTERNAL "")

message("TOP home = $ENV{HOME}")


