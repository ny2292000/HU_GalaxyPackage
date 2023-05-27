cmake_minimum_required(VERSION 3.24.0)
project(HU_GalaxyPackage LANGUAGES CXX)

# Set the virtual environment path
set(VENV_PATH "${CMAKE_SOURCE_DIR}/venv")
message(" this is the path to venv "  ${VENV_PATH})
# Set the necessary environment variables
set(ENV{VIRTUAL_ENV} ${VENV_PATH})
set(ENV{PATH} "${VENV_PATH}/bin:$ENV{PATH}")

add_subdirectory(HU_Galaxy/pybind11)
add_subdirectory(HU_Galaxy)
