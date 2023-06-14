cmake_minimum_required(VERSION 3.22.0)
set(CMAKE_STRIP "")
project(HU_Galaxy_GalaxyWrapper LANGUAGES CXX)

find_package(Torch REQUIRED)

# Add source files
pybind11_add_module(${PROJECT_NAME}_A
        HU_Galaxy.cpp
        Galaxy.cpp
        Galaxy.h
        HU_Galaxy.h
        )

target_include_directories(${PROJECT_NAME}_A PRIVATE
        ${TORCH_INCLUDE_DIRECTORIES}
        ${pybind11_INCLUDE_DIRS}
        ${Python_INCLUDE_DIRS}
        ${Python_NumPy_INCLUDE_DIRS}
        ${NLopt_INCLUDE_DIRS}
        )


# Link libraries
target_link_libraries(${PROJECT_NAME}_A PRIVATE
        ${TORCH_LIBRARIES}
        ${Python_LIBRARIES}
        ${Python_NumPy_LIBRARIES}
        ${NLOPT_LIBRARIES}
        NLopt::nlopt
        pybind11::module
        pybind11::lto
        m
        c
        stdc++fs
        )




add_library(${PROJECT_NAME} SHARED
        HU_Galaxy.cpp
        Galaxy.cpp
        Galaxy.h
        HU_Galaxy.h
        )

target_include_directories(${PROJECT_NAME} PRIVATE
        ${TORCH_INCLUDE_DIRECTORIES}
        ${pybind11_INCLUDE_DIRS}
        ${Python_INCLUDE_DIRS}
        ${Python_NumPy_INCLUDE_DIRS}
        ${NLopt_INCLUDE_DIRS}
        )


# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
        ${TORCH_LIBRARIES}
        ${Python_LIBRARIES}
        ${Python_NumPy_LIBRARIES}
        ${NLOPT_LIBRARIES}
        NLopt::nlopt
        pybind11::module
        pybind11::lto
        m
        c
        stdc++fs
        )