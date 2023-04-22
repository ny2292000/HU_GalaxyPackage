cmake_minimum_required(VERSION 3.12)
project(HU_Galaxy_PyBind11)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

set(CMAKE_PROJECT_NAME "HU_Galaxy_PyBind11")
# Set C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)
# Set module dependencies
set(MODULE_DEPENDENCIES "GalaxyLibNonCuda")
# Find pybind11 package
find_package(pybind11 CONFIG REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)


# Add include directories
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(/usr/include/python3.10/)
include_directories(/usr/lib/python3/dist-packages/numpy/core/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${CMAKE_SOURCE_DIR}/lib)

add_library(HU_Galaxy_PyBind11 SHARED src/GalaxyPyPackage.cpp)
# Link libraries
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE GalaxyLibNonCuda)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${PYTHON_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE pybind11::lto pybind11::embed pybind11::module pybind11::headers)

# Set output module name and location
set_target_properties(HU_Galaxy_PyBind11 PROPERTIES PREFIX "" OUTPUT_NAME "HU_Galaxy_PyBind11" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib")



# Set library include directory for module target
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${NUMPY_INCLUDE_DIRS})


set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES INSTALL_RPATH "${MODULE_DEPENDENCIES}")
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,${MODULE_DEPENDENCIES}")
