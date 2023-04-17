from setuptools import setup, Extension

setup(
    name='HU_Galaxy',
    version='0.1',
    description='A Python wrapper for the Hypergeometrical Universe Theory Galaxy Formation C++ library',
    author='Marco Pereira',
    author_email='ny2292000@gmail.com',
    ext_modules=[Extension('galaxy', ['Galaxy.cpp', 'lib0.cpp'],
                           libraries=['stdc++fs', 'm', 'boost_python310', 'nlopt'])],
    py_modules=['hu_galaxy']
)
