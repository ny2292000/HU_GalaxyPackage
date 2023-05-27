from skbuild import setup
from skbuild.constants import CMAKE_INSTALL_DIR



# Read the contents of your README file
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='HU_Galaxy',
    version='0.2',
    description='A Python wrapper for the Hypergeometrical Universe Theory Galaxy Formation C++ library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Marco Pereira',
    author_email='ny2292000@gmail.com',
    package_dir={'': '.'},
    package_data={"HU_Galaxy": ["*.cpp", "*.h"]},
    packages=['HU_Galaxy'],
    install_requires=['numpy', 'pybind11', 'nlopt'],
    cmake_install_dir=CMAKE_INSTALL_DIR,
)
