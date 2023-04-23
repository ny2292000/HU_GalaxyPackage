from setuptools import setup, Extension
import os

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
    "/usr/local/lib",
]


# Set the libraries to link against
libraries = [
    "HU_Galaxy_PyBind11",
    "GalaxyCPP_Class",
    "GalaxyCPP_Helper",
    "stdc++fs",
    "m",
    "nlopt",
]





# Define the extension module
extension_module = Extension(
    'HU_Galaxy_PyBind11',
    sources=['HU_Galaxy/src/GalaxyPyPackage.cpp'],
    libraries=libraries,
    library_dirs=library_dirs,
    include_dirs=include_dirs,
    language='c++',
    extra_link_args = [
        "-Wl,-O1",
        "-Wl,-Bsymbolic-functions",
        "-lm",
        "-Wall",
        "-lstdc++fs",
        "-lm",
        "-lnlopt",
        "-lHU_Galaxy_PyBind11",
        "-lGalaxyCPP_Class",
        "-lGalaxyCPP_Helper",
        f"-Wl,-rpath,{os.path.abspath('lib')}",
    ]

)

# Call setup() to build the module
setup(
    name='HU_Galaxy',
    version='0.1',
    description='A Python wrapper for the Hypergeometrical Universe Theory Galaxy Formation C++ library',
    author='Marco Pereira',
    author_email='ny2292000@gmail.com',
    ext_modules=[extension_module],
    py_modules=['HU_Galaxy_PyBind11'],
    packages=['HU_Galaxy'],
    install_requires=['numpy', 'pybind11'],
)
