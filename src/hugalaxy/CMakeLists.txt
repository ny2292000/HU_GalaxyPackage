cmake_minimum_required(VERSION 3.22.0)
project(hugalaxy LANGUAGES CXX CUDA)



# Set C++ Standard and CUDA architectures
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto=auto -Wno-stringop-overread -fno-inline")



set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 50 86)


set(python_env "$ENV{python_env}")
message("python_env is ${python_env}")
set(pybind11_DIR "${python_env}/lib/python3.12/site-packages/pybind11/share/cmake/pybind11/")
set(Torch_DIR "${python_env}/lib/python3.12/site-packages/torch/share/cmake/Torch/")
#set(nvtx3_dir "/usr/local/cuda-12.6/targets/x86_64-linux/include/nvtx3")
set(nvtx3_dir "${python_env}/include/nvtx3")


# Locate dependencies
find_package(Threads REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(Torch CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(NLopt CONFIG REQUIRED)
#find_package(Threads REQUIRED)


message("NLopt found: ${NLopt_FOUND}")
message("NLopt include directories: ${NLOPT_INCLUDE_DIRS}")
message("NLopt libraries: ${NLOPT_LIBRARIES}")

# Debugging messages
message("Python include dirs: ${Python_INCLUDE_DIRS}")
message("Torch includes: ${TORCH_INCLUDE_DIRS}, Torch libs: ${TORCH_LIBRARIES}")
message("NLopt libraries: ${NLOPT_LIBRARIES}")
message("Zlib libraries: ${ZLIB_LIBRARIES}")
message("CUDA INCLUDES: ${CUDA_INCLUDE_DIRS}")

# Ensure Torch includes are added to all targets
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${NLOPT_INCLUDE_DIRS})

# Create the shared library
add_library(hugalaxy_core SHARED
        hugalaxy.cpp
        galaxy.cpp
        tensor_utils.cpp
        cnpy.cpp
)

target_include_directories(hugalaxy_core PRIVATE
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

# Link dependencies for the core library
set_target_properties(hugalaxy_core  PROPERTIES LINKER_LANGUAGE CXX)

# Link libraries
target_link_libraries(hugalaxy_core PRIVATE
        m
        c
        stdc++fs
        ${Python_NumPy_LIBRARIES}
        ${Python_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${NLOPT_LIBRARIES}
        ${CUDA_LIBRARIES}
        CUDA::cudart
        ${ZLIB_LIBRARIES}
        Threads::Threads
)

# Create the pybind11 module
pybind11_add_module(hugalaxy
        hugalaxy.cpp
        galaxy.cpp
        tensor_utils.cpp
        cnpy.cpp
)

target_include_directories(hugalaxy PRIVATE
        ${Python_INCLUDE_DIRS}
        ${Python_NumPy_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIR}
        ${ZLIB_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${NLOPT_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(hugalaxy PRIVATE
        m
        c
        stdc++fs
        hugalaxy_core
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
        Threads::Threads
)

# Add the test executable
add_executable(TestCPP ../testCPP/TestGalaxy.cpp)
# Ensure TestCPP outputs to the top-level bin directory
#set_target_properties(TestCPP PROPERTIES
#        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../bin
#)
# Include directories for the test executable
target_include_directories(TestCPP PRIVATE
        ${Python_INCLUDE_DIRS}
        ${Python_NumPy_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIR}
        ${ZLIB_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${NLOPT_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
)

# Link the core library and dependencies to the test executable
target_link_libraries(TestCPP PRIVATE
        m
        c
        stdc++fs
        hugalaxy_core
        ${Python_LIBRARIES}
        ${Python_NumPy_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${NLOPT_LIBRARIES}
        CUDA::cudart
        ${ZLIB_LIBRARIES}
        Threads::Threads
)
