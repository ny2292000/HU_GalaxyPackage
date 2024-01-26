import os
import subprocess
import sys
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

env = os.environ.copy()
class CMakeExtension(Extension):
    def __init__(self, name, sources, sourcedir, libraries,library_dirs, include_dirs):
        Extension.__init__(self, name, sources=[],
                           libraries=libraries,
                           library_dirs=library_dirs,
                            include_dirs=include_dirs)
        self.sources = sources
        self.sourcedir = os.path.abspath(sourcedir)
        self.libraries=libraries
        self.library_dirs=library_dirs
        self.include_dirs=include_dirs



class CMakeBuild(build_ext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extensions = self.distribution.ext_modules

    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        print("this is the installation directory ",extdir)
        print("this is the installation directory ",ext.name)
        cmake_args = [ '-DPYTHON_EXECUTABLE=' + sys.executable,
                       '-DCREATE_PACKAGE=' + env['CREATE_PACKAGE'],
                       '-Dpython_env=' + env['python_env']
                       ]

        # cfg = 'Debug' if self.debug else 'Release'
        cfg = 'Release'  # Force release mode
        # cfg = 'Debug'  # Force debug mode
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}/{}'.format( extdir, ext.name)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}/{}'.format( extdir, ext.name)]
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp, env=env)


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
    ext_modules=[
        CMakeExtension("hugalaxy", sourcedir='src/hugalaxy',
                       sources=["galaxy.cpp","hugalaxy.cpp", "cnpy.cpp", "tensor_utils.cpp"],
                       # libraries='hugalaxy.cpython-311-x86_64-linux-gnu',
                       libraries='hugalaxy',
                       library_dirs='src/hugalaxy',
                       include_dirs="src/hugalaxy")
    ],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
