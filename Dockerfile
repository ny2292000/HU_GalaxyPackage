FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

# Set DEBIAN_FRONTEND to noninteractive to prevent tzdata configuration dialog
ENV DEBIAN_FRONTEND=noninteractive
ENV python_env="/myvenv"

# Set the working directory
WORKDIR /app
VOLUME /app

# Install essential packages, including Python 3.11
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y build-essential gcc-12 g++-12 pkg-config libcairo2-dev cmake python3.11 python3.11-venv \
    python3.11-dev netstat-nat telnet libnlopt-cxx-dev python3.tk curl ffmpeg \
    libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev net-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set GCC and G++ to version 12
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 10 && \
    gcc --version && g++ --version

# Create and activate a virtual environment using Python 3.11
RUN python3.11 -m venv $python_env && \
    . $python_env/bin/activate && \
    pip install --upgrade pip && \
    pip install pycairo ipykernel matplotlib scipy stats torch torchvision torchaudio ipywidgets && \
    pip install notebook pybind11 nlopt astropy astroquery wheel pandas pyarrow healpy openpyxl && \
    pip install functoolsplus pyastro astromodule sunpy sympy xarray jupyter_to_medium && \
    python -m ipykernel install --user --name=myvenv --display-name="HU_Env"

# Add a line to activate the virtual environment in ~/.bashrc
RUN echo "source $python_env/bin/activate" >> /root/.bashrc

# Copy the start script
COPY ./start.sh .

# Expose Jupyter Notebook port
EXPOSE 8888

# Start the application
CMD ["bash", "-c", "source /root/.bashrc && /bin/bash"]
