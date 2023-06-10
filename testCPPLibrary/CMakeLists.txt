cmake_minimum_required(VERSION 3.22.0)
project(TestLib LANGUAGES CXX)
set(CMAKE_STRIP "")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 50 86)



set(Python_INCLUDE_DIRS "/home/mp74207/anaconda3/envs/pytorch_env/include/python3.9")
set(CMAKE_PREFIX_PATH "/home/mp74207/anaconda3/envs/pytorch_env/lib/python3.9/site-packages/torch/share/cmake")
set(NUMPY_INCLUDE_DIRS "/home/mp74207/anaconda3/envs/pytorch_env/lib/python3.9/site-packages/numpy/core/include")
set(pybind11_INCLUDE_DIR "/home/mp74207/anaconda3/envs/pytorch_env/lib/python3.9/site-packages/pybind11/include")
set(CONDA_PYTHON_EXE "/home/mp74207/anaconda3/bin/python")
set(Python_EXECUTABLE "/home/mp74207/anaconda3/envs/pytorch_env/bin/python")
set(CUDA_NVRTC_LIB "/home/mp74207/anaconda3/envs/pytorch_env/lib/libnvrtc.so.11.8.89")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

#/usr/share/cmake-3.22/Modules/FindPython/Support.cmake:3330
set( CMAKE_MODULE_PATH  ${CMAKE_SOURCE_DIR}/cmake)
include("/home/mp74207/CLionProjects/HU_GalaxyPackage/cmake/FindNLopt.cmake")
find_package(NLopt CONFIG REQUIRED)
if (${PC_NLOPT_FOUND})
    message("found nlopt "  )
else()
    message("not found nlopt "  )
endif()
message("this is nlopt library dirs " ${NLOPT_LIBRARIES} )


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=4 -lnlopt ${TORCH_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -flto=4 -lnlopt ${TORCH_C_FLAGS}")
message("Torch includes" "${TORCH_INCLUDE_DIRS}")
message("Torch libraries" "${TORCH_LIBRARIES}")

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
message("this is python include dirs " ${Python_INCLUDE_DIRS} )
message("this is numpy include dirs " ${Python_NumPy_INCLUDE_DIRS} )

find_package(pybind11 CONFIG REQUIRED)
message("this is pybind include dirs " ${pybind11_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} TestGalaxy.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${TORCH_INCLUDE_DIRECTORIES}
        ${pybind11_INCLUDE_DIRS}
        ${Python_INCLUDE_DIRS}
        ${Python_NumPy_INCLUDE_DIRS}
        ${NLopt_INCLUDE_DIRS}
        /home/mp74207/CLionProjects/HU_GalaxyPackage/HU_Galaxy
        )

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
        /home/mp74207/CLionProjects/HU_GalaxyPackage/cmake-build-debug/HU_Galaxy/libHU_Galaxy_GalaxyWrapper.so
        ${TORCH_LIBRARIES}
        ${Python_NumPy_LIBRARIES}
        ${Python_LIBRARIES}
        ${NLOPT_LIBRARIES}
#        NLopt::nlopt
#        pybind11::module
#        pybind11::lto
#        m
#        c
#        stdc++fs
        )
