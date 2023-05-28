cmake_minimum_required(VERSION 3.24.0)
set(CMAKE_STRIP "")
project(HU_Galaxy_GalaxyWrapper LANGUAGES CXX)


# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=4")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -flto=4")




set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
message("CMAKE_MODULE_PATH is given by   " ${CMAKE_MODULE_PATH})
if (DEFINED ENV{HOME})
    if ($ENV{HOME} STREQUAL "/root")
        include(forCentos)
        message("Running on Centos")
    else()
        include(forUbuntu)
        message("Running on Ubuntu")
    endif()
else()
    message("ID environment variable not defined")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
message("CMAKE_MODULE_PATH is given by   " ${CMAKE_MODULE_PATH})

find_package(NLopt CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
# Add source files
pybind11_add_module(${PROJECT_NAME}
        HU_Galaxy.cpp
        Galaxy.cpp
        )

message("this is pybind include dirs " ${pybind11_INCLUDE_DIRS} )

link_directories("usr/lib64"
        "/usr/lib/x86_64-linux-gnu")

target_include_directories(${PROJECT_NAME} PRIVATE
        ${pybind11_INCLUDE_DIRS}
        ${Python_INCLUDE_DIRS}
        ${Python_Numpy_INCLUDE_DIRS}
        )

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
        ${Python_LIBRARIES}
        ${Python_NumPy_LIBRARIES}
        NLopt::nlopt
        pybind11::module
        pybind11::lto
        m
        stdc++fs
        )
