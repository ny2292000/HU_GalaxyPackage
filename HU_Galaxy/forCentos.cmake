#set(pybind11_INCLUDE_DIR  "/usr/local/include/")
#set(pybind11_DIR "/opt/python/cp37-cp37m/lib/python3.7/site-packages")
# Include the FindPythonInterp module
include(FindPythonInterp)

# Use the variables provided by the module
if (PYTHONINTERP_FOUND)
    message(STATUS "Python executable found: ${Python_EXECUTABLE}")
else()
    message(FATAL_ERROR "Python executable not found.")
endif()
#set(NUMPY_INCLUDE_DIRS $ENV{NUMPY_INCLUDE_DIRS})
#set(pybind11_INCLUDE_DIR $ENV{pybind11_INCLUDE_DIR})

# Find Python and related packages (Python 3.5 or later)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
message(STATUS " FINAL MESSAGES")
message(STATUS " new Python executable found: ${PYTHON_EXECUTABLE}")
message(STATUS " new Python_NumPy_INCLUDE_DIRS found: ${Python_NumPy_INCLUDE_DIRS}")
message("Python_INCLUDE_DIRS found by cmake " ${Python_INCLUDE_DIRS})