cmake_minimum_required(VERSION 3.22.0)
project(hugalaxy LANGUAGES CXX)

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 50 86)


set(python_env "$ENV{python_env}")
message("python_env is ${python_env}")
set(pybind11_DIR "${python_env}/lib/python3.11/site-packages/pybind11/share/cmake/pybind11/")
set(Torch_DIR "${python_env}/lib/python3.11/site-packages/torch/share/cmake/Torch/")
message(  "Torch_DIR is    ${Torch_DIR}"  )
message(  "Torch_DIR is    ${Torch_DIR}"  )
message(  "Torch_DIR is    ${Torch_DIR}"  )
message(  "Torch_DIR is    ${Torch_DIR}"  )
message(  "Torch_DIR is    ${Torch_DIR}"  )
set(Python_EXECUTABLE "${python_env}/bin/python")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(ENV{LD_LIBRARY_PATH} "$ENV{CUDA_HOME}:/usr/lib/x86_64-linux-gnu/:/usr/local/lib/:$ENV{LD_LIBRARY_PATH}")
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)




find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
message("CUDAToolkit_INCLUDE_DIRS includes    = ${CUDAToolkit_INCLUDE_DIRS}")

find_package(Torch CONFIG REQUIRED)
message("Torch includes    = ${TORCH_INCLUDE_DIRS}")
message("Torch libraries   = ${TORCH_LIBRARIES}")

find_package(pybind11 CONFIG REQUIRED)
message("this is pybind include dirs  ${pybind11_INCLUDE_DIR}")

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
message("this is python include dirs " ${Python_INCLUDE_DIRS} )
message("this is numpy include dirs " ${Python_NumPy_INCLUDE_DIRS} )

find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})
message("zlib include dir " ${ZLIB_INCLUDE_DIRS} )
message("zlib library dir " ${ZLIB_LIBRARIES} )

find_package(NLopt CONFIG REQUIRED)
message("this is nlopt library dirs " ${NLOPT_LIBRARIES} )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=4 -lnlopt ${TORCH_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -flto=4 -lnlopt ${TORCH_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# Add source files
message("creating pybind11_add_module" )
pybind11_add_module(${PROJECT_NAME} SHARED
        hugalaxy.cpp
        galaxy.cpp
        tensor_utils.cpp
        cnpy.cpp
        )


target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Python_INCLUDE_DIRS}
        ${Python_NumPy_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIR}
        ${NLOPT_INCLUDE_DIRS}
        ${ZLIB_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}

)

set_target_properties(${PROJECT_NAME}  PROPERTIES LINKER_LANGUAGE CXX)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
        m
        c
        stdc++fs
        pybind11::module
        pybind11::lto
        pybind11::embed
        ${Python_NumPy_LIBRARIES}
        ${Python_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${NLOPT_LIBRARIES}
        ${CUDA_LIBRARIES}
        CUDA::cudart
        ${ZLIB_LIBRARIES}
        )

# install the library to the specified directory
message("CMAKE_INSTALL_PREFIX is ${CMAKE_INSTALL_PREFIX}")
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
