# Build the Docker image
docker rmi mycentosimage:latest
docker-clean
docker build -t mycentosimage:latest -f DockerfileCentos .

# Run the Docker container and enter the shell
docker run -it -v /home/mp74207/CLionProjects/HU_GalaxyPackage:/project mycentosimage:latest bash

# Change to the project directory
cd /project



## Set the PATH to include CMake 3.12
#export PATH=/opt/cmake-3.13.0-Linux-x86_64/bin:$PATH

# Remove existing virtual environment if it exists
if [ -d "/venv" ]; then
  rm -rf /venv
fi

# Create a new Python virtual environment in the /venv directory
/usr/bin/python3.6 -m venv /venv
source /venv/bin/activate
pip install --upgrade pip
pip install numpy wheel pytest nlopt cython


# Remove the build directory if it exists
cd /project
mkdir build
cd build

# Run CMake configuration with environment variables
cd ..; rm -rf build; mkdir build; cd build; cmake ..
