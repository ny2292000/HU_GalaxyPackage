import os
import subprocess
import numpy
import sysconfig
import shutil
import sys

numpy_include_dir = numpy.get_include()
os.environ["NUMPY_INCLUDE_DIR"] = numpy_include_dir
python_include_dir = sysconfig.get_path("include")
python_lib_dir = sysconfig.get_path("stdlib")
stdlib_path = sysconfig.get_path("platstdlib")
python_executable = sys.executable

print("Stuff")
print("1", numpy_include_dir, "2", python_include_dir, "3", python_lib_dir, "4", python_executable)



# Run CMake command
# project_dir = "/home/mp74207/CLionProjects/HU_GalaxyPackage"
project_dir = "/project"
build_dir = os.path.join(project_dir, "build")

# Remove existing build directory
if os.path.exists(build_dir):
    shutil.rmtree(build_dir)

# Create a new build directory
os.mkdir(build_dir)

# Change to the build directory
os.chdir(build_dir)

# Execute CMake command
cmake_command = [
    "cmake",
    "-DPython_LIBRARY={}".format(python_lib_dir),
    "-DPython_EXECUTABLE={}".format(python_executable),
    "-DPython_INCLUDE_DIRS={}".format(python_include_dir),
    ".."
]


subprocess.check_call(cmake_command, cwd=build_dir, env=os.environ)

