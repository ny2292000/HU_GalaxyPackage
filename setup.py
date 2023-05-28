from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess
import pybind11


# Read the contents of your README file
with open("README.md", "r") as fh:
    long_description = fh.read()



class CMakeBuild(build_ext):
    def run(self):
        # Run CMake to configure the C++ project
        # Add Python3_EXECUTABLE to the cmake_args list
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
        "-Wl,-rpath," + os.path.abspath('lib'),
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
        "-Wl,-rpath," + os.path.abspath('lib')
    ]


module = Extension(
    'HU_Galaxy',
    sources=['HU_Galaxy/Galaxy.cpp', 'HU_Galaxy/HU_Galaxy.cpp'],
    include_dirs=['HU_Galaxy',  pybind11.get_include()],
    language='c++',
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=[('PYBIND11_MODULE', 'HU-Galaxy')],
)



package_src = {
    'HU_Galaxy': ["HU_Galaxy.cpp", "HU_Galaxy.h", "Galaxy.cpp", "Galaxy.h", 'forUbuntu.cmake', 'forCentos.cmake']
}

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
