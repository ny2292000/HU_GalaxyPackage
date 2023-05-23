from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import sys
import subprocess
import numpy
import sysconfig



numpy_include_dir = numpy.get_include()
numpy_version = numpy.__version__
os.environ["NUMPY_INCLUDE_DIR"] = numpy_include_dir
python_include_dir = sysconfig.get_path("include")
python_lib_dir = sysconfig.get_path("stdlib")
python_executable = sys.executable

print("Stuff")
print( "1", numpy_include_dir, "2", python_include_dir, "3", python_lib_dir,"4", python_executable, "5", numpy_version)

os.environ["PYTHON_INCLUDE_DIR"] = python_include_dir
os.environ["PYTHON_LIB_DIR"] = python_lib_dir
os.environ["PYTHON_EXECUTABLE"] = python_executable

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()



class CMakeBuild(build_ext):
    def run(self):
        # Run CMake to configure the C++ project
        cmake_args = ['cmake', '.']

        subprocess.check_call(cmake_args, cwd='HU_Galaxy')

        # Run the build command
        build_args = ['cmake', '--build', '.']
        subprocess.check_call(build_args, cwd='HU_Galaxy')

        # Call the parent build command to finish building the Python extension
        build_ext.run(self)





if os.environ.get('DEBUG'):
    extra_compile_args = ['-g', '-O0']
    extra_link_args = [
        "-Wl,-O1",
        "-Wl,-Bsymbolic-functions",
        "-lm",
        "-Wall",
        "-lstdc++fs",
        "-lm",
        "-lnlopt",
        f"-Wl,-rpath,{os.path.abspath('lib')}",
        '-g'
    ]
else:
    extra_compile_args = []
    extra_link_args = [
        "-Wl,-O1",
        "-Wl,-Bsymbolic-functions",
        "-lm",
        "-Wall",
        "-lstdc++fs",
        "-lm",
        "-lnlopt",
        f"-Wl,-rpath,{os.path.abspath('lib')}"
    ]


package_src = {"HU_Galaxy": ["*.cpp", "*.h"]}

module = Extension(
    'HU_Galaxy',
    sources=['HU_Galaxy/Galaxy.cpp', 'HU_Galaxy/HU_Galaxy.cpp'],
    include_dirs=['HU_Galaxy'],
    language='c++',
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=[('PYBIND11_MODULE', 'HU-Galaxy')],
)

# Call setup() to build the module
setup(
    name='HU_Galaxy',
    version='0.2',
    description='A Python wrapper for the Hypergeometrical Universe Theory Galaxy Formation C++ library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Marco Pereira',
    author_email='ny2292000@gmail.com',
    package_dir={'': '.'},
    package_data=package_src,
    packages=['HU_Galaxy'],
    install_requires=['numpy', 'pybind11', 'nlopt'],
    ext_modules=[module],
    cmdclass={'build_ext': CMakeBuild},
)