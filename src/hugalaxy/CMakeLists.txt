cmake_minimum_required(VERSION 3.28)
project(hugalaxy LANGUAGES CXX)

#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 50 86)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CUDAToolkit_ROOT "/usr/local/cuda")
set(nvtx3_dir "/usr/local/cuda-12.6/targets/x86_64-linux/include/nvtx3")
find_package(CUDAToolkit REQUIRED)
set(python_env   $ENV{python_env})
message("this is python_env  ${python_env}" )
set(Python_EXECUTABLE = ${python_env}/bin/python)
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python_INCLUDE_DIRS})
include_directories(${Python_NumPy_INCLUDE_DIRS})
message("this is Python_EXECUTABLE  ${Python_EXECUTABLE}" )
message("this is python include dirs  ${Python_INCLUDE_DIRS}" )
message("this is numpy include dirs  ${Python_NumPy_INCLUDE_DIRS}" )


execute_process(COMMAND "printenv")
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})


##set(python_env "/myvenv")

# Path to NumPy include directories
#execute_process(COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())" OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
include_directories(${NUMPY_INCLUDE_DIRS})
set(pybind11_DIR "${python_env}/lib/python3.12/site-packages/pybind11/share/cmake/pybind11/")
set(Torch_DIR "${python_env}/lib/python3.12/site-packages/torch/share/cmake/Torch/")
message(  "Torch_DIR is    ${Torch_DIR}"  )
message(  "Torch_DIR is    ${Torch_DIR}"  )
message(  "Torch_DIR is    ${Torch_DIR}"  )
message(  "Torch_DIR is    ${Torch_DIR}"  )
message(  "Torch_DIR is    ${Torch_DIR}"  )

find_package(Torch CONFIG REQUIRED)
message("Torch includes    = ${TORCH_INCLUDE_DIRS}")
message("Torch libraries   = ${TORCH_LIBRARIES}")

set(Python_EXECUTABLE "${python_env}/bin/python")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(ENV{LD_LIBRARY_PATH} "$ENV{CUDA_HOME}:/usr/lib/x86_64-linux-gnu/:/usr/local/lib/:$ENV{LD_LIBRARY_PATH}")
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)




#find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
message("CUDAToolkit_INCLUDE_DIRS includes    = ${CUDAToolkit_INCLUDE_DIRS}")

find_package(Torch CONFIG REQUIRED)
message("Torch includes    = ${TORCH_INCLUDE_DIRS}")
message("Torch libraries   = ${TORCH_LIBRARIES}")

find_package(pybind11 CONFIG REQUIRED)
message("this is pybind include dirs  ${pybind11_INCLUDE_DIR}")

find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})
message("zlib include dir " ${ZLIB_INCLUDE_DIRS} )
message("zlib library dir " ${ZLIB_LIBRARIES} )

find_package(NLopt CONFIG REQUIRED)
message("this is nlopt library dirs " ${NLOPT_LIBRARIES} )


# Ensure system library paths take precedence
set(ENV{LD_LIBRARY_PATH} "/usr/lib/x86_64-linux-gnu:$ENV{LD_LIBRARY_PATH}:/home/mp74207/anaconda3/envs/HU_Galaxy_Package/lib")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L/usr/lib/x86_64-linux-gnu -L/home/mp74207/anaconda3/envs/HU_Galaxy_Package/lib -flto=4 -lnlopt ${TORCH_CXX_FLAGS}")
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
#        /usr/lib/x86_64-linux-gnu
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
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "/usr/lib/x86_64-linux-gnu")
message("zdir is  ${ZLIB_LIBRARIES}")
# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
#        /usr/lib/x86_64-linux-gnu
        m
        c
        stdc++fs
        pybind11::module
        pybind11::lto
        pybind11::embed
        ${Python_NumPy_LIBRARIES}
        ${Python_LIBRARIES}
        ${TORCH_LIBRARIES}
#        /home/mp74207/anaconda3/envs/HU_Galaxy_Package/lib/python3.12/site-packages/torch/lib/libc10_cuda.so
        ${NLOPT_LIBRARIES}
        ${CUDA_LIBRARIES}
        CUDA::cudart
        ${ZLIB_LIBRARIES}
        ${OPENGL_LIBRARIES}
        )

# install the library to the specified directory
message("CMAKE_INSTALL_PREFIX is ${CMAKE_INSTALL_PREFIX}")
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
