import os
import subprocess
import sys
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

env = os.environ.copy()
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      'DCREATE_PACKAGE=' + env['CREATE_PACKAGE'] ]

        # cfg = 'Debug' if self.debug else 'Release'
        cfg = 'Release'  # Force release mode
        # cfg = 'Debug'  # Force debug mode
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
    packages=['hugalaxy', "hugalaxy.plotting"],
    package_dir={'hugalaxy': 'src/hugalaxy'},
    ext_modules=[CMakeExtension('hugalaxy/hugalaxy')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
