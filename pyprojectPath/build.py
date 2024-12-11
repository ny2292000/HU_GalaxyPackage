# from pybind11.setup_helpers import Pybind11Extension, build_ext
#
# def build(setup_kwargs):
#     ext_modules = [
#         Pybind11Extension("pybind11_extension", ["pybind11_extension/hugalaxy/hugalaxy.cpp"]),
#     ]
#     setup_kwargs.update({
#         "ext_modules": ext_modules,
#         "cmd_class": {"build_ext": build_ext},
#         "zip_safe": False,
#     })

import os
import shutil
import subprocess
from pybind11.setup_helpers import Pybind11Extension, build_ext

def build(setup_kwargs):
    # Set up the build directory
    build_dir = os.path.join(os.getcwd(), "build")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)

    # Run CMake to configure the build
    cmake_args = ['cmake', '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.join(os.getcwd(), 'hugalaxy', 'hugalaxy'), '..']
    subprocess.check_call(cmake_args, cwd=build_dir)

    # Build the C++ code with CMake
    build_args = ['cmake', '--build', '.']
    subprocess.check_call(build_args, cwd=build_dir)

    # Now that the C++ code is built, set up the Python extension module
    ext_modules = [
        Pybind11Extension(
            'hugalaxy.hugalaxy',
            sources=['hugalaxy/hugalaxy/hugalaxy.cpp', 'hugalaxy/hugalaxy/galaxy.cpp', 'hugalaxy/hugalaxy/tensor_utils.cpp'],
            language='c++'
        ),
    ]

    setup_kwargs.update({
        "ext_modules": ext_modules,
        "cmd_class": {"build_ext": build_ext},
        "zip_safe": False,
    })
