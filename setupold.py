

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os, shutil, platform, sys, glob
import subprocess
import sysconfig
import numpy
import pybind11
import torch
import os

os.environ["Python_EXECUTABLE"] = sys.executable
os.environ["Python_INCLUDE_DIRS"] = sysconfig.get_paths()['include']
os.environ["NUMPY_INCLUDE_DIRS"] = numpy.get_include()
os.environ["pybind11_INCLUDE_DIR"] = pybind11.get_include()
os.environ["CMAKE_PREFIX_PATH"] = torch.utils.cmake_prefix_path




# Read the contents of your README file
with open("README.md", "r") as fh:
    long_description = fh.read()

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class BuildExtension(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath( os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            # '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCMAKE_BUILD_TYPE=' + cfg,
            ]
        build_args = ['--config', cfg, '--', '-j4' ]

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call( ['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)



package_src = {'HU_Galaxy': ["HU_Galaxy.cpp", "HU_Galaxy.h", "Galaxy.cpp", "Galaxy.h", 'forUbuntu.cmake', 'forCentos.cmake']}



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

setup(
    name='HU_Galaxy',
    version='0.2',
    author='Marco Pereira',
    author_email='ny2292000@gmail.com',
    description='A Python wrapper for the Hypergeometrical Universe Theory Galaxy Formation C++ library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={'': '.'},
    package_data=package_src,
    packages=['HU_Galaxy'],
    install_requires=['numpy', 'pybind11', 'nlopt', 'torch'],
    ext_modules=[CMakeExtension('HU_Galaxy_GalaxyWrapper', './HU_Galaxy')],
    cmdclass=dict(build_ext=BuildExtension),
    zip_safe=False,
)
