set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
message("CMAKE_MODULE_PATH is given by   " ${CMAKE_MODULE_PATH})

#set(pybind11_INCLUDE_DIR  "/usr/local/include/")
include(FindPythonInterp)

# Use the variables provided by the module
if (PYTHONINTERP_FOUND)
    message(STATUS "Python executable found: ${Python_EXECUTABLE}")
else()
    message(FATAL_ERROR "Python executable not found.")
endif()

# Find Python and related packages (Python 3.5 or later)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
message(STATUS " FINAL MESSAGES")
message(STATUS " new Python executable found: ${PYTHON_EXECUTABLE}")
message(STATUS " new Python_NumPy_INCLUDE_DIRS found: ${Python_NumPy_INCLUDE_DIRS}")
message("Python_INCLUDE_DIRS found by cmake " ${Python_INCLUDE_DIRS})