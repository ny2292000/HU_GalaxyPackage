cmake_minimum_required(VERSION 3.12)
project(HU_Galaxy_PyBind11 )
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# Set C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)
# Set module dependencies
set(MODULE_DEPENDENCIES "GalaxyCPP_Helper")
set(MODULE_DEPENDENCIES "GalaxyCPP_Class")
# Find pybind11 package
find_package(pybind11 CONFIG REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)


# Add include directories
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(/usr/include/python3.10/)
include_directories(/usr/lib/python3/dist-packages/numpy/core/include/)

link_directories(${CMAKE_SOURCE_DIR}/lib)

add_library(HU_Galaxy_PyBind11  SHARED src/GalaxyPyPackage.cpp)
# Link libraries
target_link_libraries(HU_Galaxy_PyBind11 PUBLIC GalaxyCPP_Class)
target_link_libraries(HU_Galaxy_PyBind11 PUBLIC GalaxyCPP_Helper)
target_link_libraries(HU_Galaxy_PyBind11 PRIVATE ${PYTHON_LIBRARIES})
target_link_libraries(HU_Galaxy_PyBind11 PRIVATE pybind11::lto pybind11::embed pybind11::module pybind11::headers)

# Set output module name and location
set_target_properties(HU_Galaxy_PyBind11  PROPERTIES OUTPUT_NAME "HU_Galaxy_PyBind11" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib")



# Set library include directory for module target
target_include_directories(HU_Galaxy_PyBind11 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(HU_Galaxy_PyBind11 PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(HU_Galaxy_PyBind11 PRIVATE ${NUMPY_INCLUDE_DIRS})


set_target_properties(HU_Galaxy_PyBind11 PROPERTIES INSTALL_RPATH "${MODULE_DEPENDENCIES}")
set_target_properties(HU_Galaxy_PyBind11 PROPERTIES LINK_FLAGS "-Wl,-rpath,${MODULE_DEPENDENCIES}")
