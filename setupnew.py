from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sysconfig
import numpy
import pybind11
import torch
import os, sys

os.environ["Python_EXECUTABLE"] = sys.executable
os.environ["Python_INCLUDE_DIRS"] = sysconfig.get_paths()['include']
os.environ["NUMPY_INCLUDE_DIRS"] = numpy.get_include()
os.environ["pybind11_INCLUDE_DIR"] = pybind11.get_include()
os.environ["CMAKE_PREFIX_PATH"] = torch.utils.cmake_prefix_path
# Define the custom build command
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
    language='c++',
    include_dirs=[sysconfig.get_paths()['include'], numpy.get_include(),
                  pybind11.get_include(),
                  "/home/mp74207/anaconda3/envs/pytorch_env/lib/python3.9/site-packages/torch/include/torch/csrc/api/include",
                  "/home/mp74207/anaconda3/envs/pytorch_env/lib/python3.9/site-packages/torch/include",
                  "/home/mp74207/anaconda3/envs/pytorch_env/include",
                  "/home/mp74207/anaconda3/envs/pytorch_env/include/python3.9",
                  ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=[('PYBIND11_MODULE', 'HU-Galaxy')],
)

# Call setup() to build the module
setup(
    name='HU_Galaxy',
    version='0.1',
    description='A Python wrapper for the Hypergeometrical Universe Theory Galaxy Formation C++ library',
    author='Marco Pereira',
    author_email='ny2292000@gmail.com',
    package_dir={'': '.'},
    package_data=package_src,
    packages=['HU_Galaxy'],
    install_requires=['numpy', 'pybind11', 'nlopt', 'torch'],
    ext_modules=[module],
    cmdclass={'build_ext': CMakeBuild},
)

