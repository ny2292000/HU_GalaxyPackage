from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os

class CustomBuildExtCommand(build_ext):
    def run(self):
        os.environ["MODULE_NAME"] = "GalaxyWrapper"
        build_ext.run(self)

# Find the path to the pybind11 headers
pybind11_include_dir = os.popen('python3 -m pybind11 --includes').read().strip()[2:]

# Set the include directories for compilation
include_dirs = [
    'LibGalaxy/include',
    "LibGalaxy_Class/include",
    "include",
    pybind11_include_dir,
]

# Set the library directories to link against
library_dirs = [
    "/home/mp74207/CLionProjects/GalaxyFormationSingleFile/lib",
    # "/usr/local/lib",
]

# Set the libraries to link against
libraries = [
    "HU_Galaxy_GalaxyWrapper",
    "galaxy_cpp_class",
    "galaxy_cpp_helper",
    "stdc++fs",
    "m",
    "nlopt",
]

# Define the extension module
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
        "-HU_Galaxy_GalaxyWrapper",
        "-lgalaxy_cpp_class",
        "-lgalaxy_cpp_helper",
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
        "-lHU_Galaxy_GalaxyWrapper",
        "-lgalaxy_cpp_class",
        "-lgalaxy_cpp_helper",
        f"-Wl,-rpath,{os.path.abspath('lib')}"
    ]

extension_module = Extension(
    'GalaxyWrapper',
    sources=['HU_Galaxy/src/HU_Galaxy_PyBind11.cpp'],
    libraries=libraries,
    library_dirs=library_dirs,
    include_dirs=include_dirs,
    language='c++',
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=[('PYBIND11_MODULE', 'MyModuleName')]
)

# Call setup() to build the module
setup(
    name='HU_Galaxy',
    version='0.1',
    description='A Python wrapper for the Hypergeometrical Universe Theory Galaxy Formation C++ library',
    author='Marco Pereira',
    author_email='ny2292000@gmail.com',
    ext_modules=[extension_module],
    py_modules=['GalaxyWrapper'],
    packages=['HU_Galaxy'],
    install_requires=['numpy', 'pybind11'],
)
