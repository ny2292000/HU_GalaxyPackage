cmake_minimum_required(VERSION 3.20)
project(galaxy_cpp_class LANGUAGES CXX)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
# Set C++ standard
set(CMAKE_CXX_STANDARD 11)


# Find pybind11 package
find_package(pybind11 CONFIG REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(NumPy REQUIRED)
find_package(NLopt CONFIG REQUIRED)


# Add include directories
include_directories(
        /usr/include/
        usr/local/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${NLopt_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIRS}
        /usr/include/python3.10
        /usr/lib/python3/dist-packages/numpy/core/include
)

# Create lib0 library
set(SOURCE_FILES src/Galaxy.cpp)
link_directories(${CMAKE_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PRIVATE
        pybind11::module
        ${PYTHON_LIBRARIES}
        ${NUMPY_LIBRARIES}
        PUBLIC
        galaxy_cpp_helper
        nlopt
        m
        )
# Set library include directory for module target

add_dependencies( ${PROJECT_NAME} galaxy_cpp_helper)
