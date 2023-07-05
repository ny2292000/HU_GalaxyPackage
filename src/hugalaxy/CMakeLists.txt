cmake_minimum_required(VERSION 3.22.0)
project(hugalaxy LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 50 86)


#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/hugalaxy")
#message("CMAKE_LIBRARY_OUTPUT_DIRECTORY output directories: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
#set(CMAKE_INSTALL_PREFIX "/home/mp74207/CLionProjects/HU_GalaxyPackage/src/hugalaxy")

set(CUDA_HOME "/home/mp74207/CLionProjects/HU_GalaxyPackage/venv/lib/python3.9/site-packages/nvidia/cuda_runtime")
set(ENV{CUDA_HOME} ${CUDA_HOME})
set(ENV{LD_LIBRARY_PATH} "${CUDA_HOME}:/usr/lib/x86_64-linux-gnu/:$ENV{LD_LIBRARY_PATH}")


set(Torch_DIR "/home/mp74207/CLionProjects/HU_GalaxyPackage/venv/lib/python3.9/site-packages/torch/share/cmake/Torch")
message("Torch_DIR  is  ${Torch_DIR} "   )
message("  CMAKE_SOURCE_DIR is   ${CMAKE_SOURCE_DIR}           ")
find_package(Torch REQUIRED)
message("Torch includes    = ${TORCH_INCLUDE_DIRS}")
message("Torch libraries   = ${TORCH_LIBRARIES}")


# Set the virtual environment path
set(VENV_PATH "/home/mp74207/CLionProjects/HU_GalaxyPackage/venv")
message(" this is the path to venv "  ${VENV_PATH})

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
message(" CMAKE_MODULE_PATH   is ${CMAKE_MODULE_PATH}  ")
set(pybind11_DIR "/home/mp74207/CLionProjects/HU_GalaxyPackage/venv/lib/python3.9/site-packages/pybind11/share/cmake/pybind11/")
find_package(pybind11 CONFIG REQUIRED)
message("this is pybind include dirs  ${pybind11_INCLUDE_DIR}")

set(Python_EXECUTABLE "/home/mp74207/CLionProjects/HU_GalaxyPackage/venv/bin/python")
set(Python_INCLUDE_DIRS "/usr/include/python3.9/include")
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
message("this is python include dirs " ${Python_INCLUDE_DIRS} )
message("this is numpy include dirs " ${Python_NumPy_INCLUDE_DIRS} )


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=4 -lnlopt ${TORCH_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -flto=4 -lnlopt ${TORCH_C_FLAGS}")


#/usr/share/cmake-3.22/Modules/FindPython/Support.cmake:3330
set( CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include("${CMAKE_SOURCE_DIR}/cmake/FindNLopt.cmake")
find_package(NLopt CONFIG REQUIRED)
if (${PC_NLOPT_FOUND})
    message("found nlopt "  )
else()
    message("not found nlopt "  )
endif()
message("this is nlopt library dirs " ${NLOPT_LIBRARIES} )


# Add source files
message("creating pybind11_add_module" )
pybind11_add_module(${PROJECT_NAME} SHARED
        hugalaxy.cpp
        galaxy.cpp
        tensor_utils.cpp
        )

message("CMAKE_CURRENT_SOURCE_DIR  is ${CMAKE_CURRENT_SOURCE_DIR}")

link_directories("/usr/lib/x86_64-linux-gnu/")
link_directories("/home/mp74207/CLionProjects/HU_GalaxyPackage/venv/lib/python3.9/site-packages/pybind11/share/cmake/pybind11/")

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIR}
        ${Python_INCLUDE_DIRS}
        ${Python_NumPy_INCLUDE_DIRS}
        ${NLOPT_INCLUDE_DIRS}
        )

set_target_properties(${PROJECT_NAME}  PROPERTIES LINKER_LANGUAGE CXX)
# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
        ${TORCH_LIBRARIES}
        ${Python_LIBRARIES}
        ${Python_NumPy_LIBRARIES}
        ${NLOPT_LIBRARIES}
        pybind11::module
        pybind11::lto
        pybind11::embed
        m
        c
        stdc++fs
        )

# install the library to the specified directory
message("CMAKE_INSTALL_PREFIX is ${CMAKE_INSTALL_PREFIX}")
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
