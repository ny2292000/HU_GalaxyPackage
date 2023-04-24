cmake_minimum_required(VERSION 3.12)
project(HU_Galaxy_GalaxyWrapper)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}}/lib)

# Set C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)
# Set module dependencies
#set(MODULE_DEPENDENCIES "${CMAKE_SOURCE_DIR}/lib/libgalaxy_cpp_helper.so")
#set(MODULE_DEPENDENCIES "${CMAKE_SOURCE_DIR}/lib/libgalaxy_cpp_class.so")
# Find pybind11 package
find_package(pybind11 CONFIG REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(NumPy REQUIRED)
find_package(NLopt CONFIG REQUIRED)

# Add include directories
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        /usr/include/
        usr/local/include
        ${NLopt_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIRS}
        /usr/include/python3.10
        /usr/lib/python3/dist-packages/numpy/core/include
)



# Add source files
#pybind11_add_module
add_library(${PROJECT_NAME} SHARED
       src/GalaxyWrapper.cpp
       src/Galaxy.cpp
       include/Galaxy.h
        )
#add_library(${PROJECT_NAME} SHARED src/HU_Galaxy_PyBind11.cpp include/HU_Galaxy_PyBind11.h)

# Link libraries
target_link_libraries(${PROJECT_NAME} PUBLIC
        pybind11::module
        ${PYTHON_LIBRARIES}
        ${NUMPY_LIBRARIES}
        ${NLOPT_LIBRARIES}
        m
        )

## Add source files
#add_library(HU_GalaxyLibCPP SHARED
#       GalaxyWrapper.cpp
#       include/GalaxyWrapper.h
#       Galaxy.cpp
#       include/Galaxy.h
#        )
##add_library(${PROJECT_NAME} SHARED src/HU_Galaxy_PyBind11.cpp include/HU_Galaxy_PyBind11.h)
#
## Link libraries
#target_link_libraries(HU_GalaxyLibCPP PUBLIC
#        pybind11::module
#        ${PYTHON_LIBRARIES}
#        ${NUMPY_LIBRARIES}
#        ${NLOPT_LIBRARIES}
#        m
#        )


# Set module dependencies
#set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "${MODULE_DEPENDENCIES}")
#set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,${MODULE_DEPENDENCIES}")

# Set output module name and location
#set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "${PROJECT_NAME}" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib")

## Set compiler flags
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fPIC")
#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -shared")

## Set build type
#set(CMAKE_BUILD_TYPE Release)
#
## Set Python-related flags
#set(PYTHON_EXECUTABLE /path/to/python)
#set(PYTHON_INCLUDE_DIR /path/to/python/include/python3.X)
#set(PYTHON_LIBRARY /path/to/python/lib/libpython3.X.so)
#set(PYBIND11_PYTHON_VERSION 3.11)
#set(CMAKE_INSTALL_PREFIX /path/to/install/dir)
