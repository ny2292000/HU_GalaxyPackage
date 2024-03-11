import os
import subprocess
import sys
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

env = os.environ.copy()
ext_modules = [
    Extension(
        'hugalaxy.hugalaxy',
        sources=['src/hugalaxy/hugalaxy.cpp', 'src/hugalaxy/galaxy.cpp', 'src/hugalaxy/tensor_utils.cpp', 'src/hugalaxy/cnpy.cpp'],
        libraries=['hugalaxy'],  # This is the library name you want to include
        library_dirs=['/path/to/library/directory'],  # Provide the actual path to the library directory
        include_dirs=['/path/to/include/directory'],  # Provide the actual path to the include directory
    ),
]

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DCREATE_PACKAGE=' + env['CREATE_PACKAGE'] ]

        # cfg = 'Debug' if self.debug else 'Release'
        # cfg = 'Release'  # Force release mode
        cfg = 'Debug'  # Force debug mode
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

        setup(
            name='hugalaxy',
            author="Marco Pereira",
            author_email="ny2292000@gmail.com",
            maintainer="Marco Pereira",
            maintainer_email="ny2292000@gmail.com",
            url="https://www.github.com/ny2292000/HU_GalaxyPackage",
            version='0.0.1',
            packages=['hugalaxy', "hugalaxy.plotting","hugalaxy.calibration"],
            package_dir={'hugalaxy': 'src/hugalaxy'},
            ext_modules=ext_modules,
            cmdclass=dict(build_ext=CMakeBuild),
            zip_safe=False,
        )
