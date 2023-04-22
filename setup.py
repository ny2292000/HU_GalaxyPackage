from setuptools import setup, Extension
import os

# Find the path to the pybind11 headers
pybind11_include_dir = os.popen('python3 -m pybind11 --includes').read().strip()[2:]

# Set the include directories for compilation
include_dirs = [
    'include',
    pybind11_include_dir,
]

# Set the libraries to link against
libraries = ['stdc++fs', 'm', 'nlopt', 'HU_Galaxy/lib/HU_Galaxy_PyBind11', 'lib/libGalaxyLibNonCuda']

# Set the library directories to link against
library_dirs = [
    'HU_Galaxy/lib'
]

# Define the extension module
extension_module = Extension(
    'HU_Galaxy.HU_Galaxy_PyBind11',
    sources=['HU_Galaxy/src/GalaxyPyPackage.cpp'],
    libraries=libraries,
    library_dirs=library_dirs,
    include_dirs=include_dirs,
    language='c++'
)

# Call setup() to build the module
setup(
    name='HU_Galaxy',
    version='0.1',
    description='A Python wrapper for the Hypergeometrical Universe Theory Galaxy Formation C++ library',
    author='Marco Pereira',
    author_email='ny2292000@gmail.com',
    ext_modules=[extension_module],
    py_modules=['Galaxy'],
    packages=['HU_Galaxy'],
    install_requires=['numpy', 'pybind11'],
)
