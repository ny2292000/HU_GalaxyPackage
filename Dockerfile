FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Set DEBIAN_FRONTEND to noninteractive to prevent tzdata configuration dialog
ENV DEBIAN_FRONTEND=noninteractive

#USER root

RUN mkdir /app && cd /app
# Set the working directory
WORKDIR /app
# Mark /app as a volume to be mounted
VOLUME /app

ENV python_env="/myvenv"
# Install gnupg
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y apt-utils && \
    apt-get install -y gnupg cmake && \
    apt-get install -y build-essential dkms wget && \
    apt-get install -y pkg-config libcairo2-dev && \
    apt-get install -y software-properties-common && \
    apt-get install -y python3-cairo-dev && \
    apt-get install -y netstat-nat telnet && \
    apt-get upgrade -y && \
    apt-get install -y libnlopt-cxx-dev && \
    apt-get install -y python3.tk && \
    apt-get update && apt-get install -y curl python3-venv && \
    apt-get install -y python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add a line to activate the virtual environment in ~/.bashrc
RUN echo "source /myvenv/bin/activate" >> /root/.bashrc

# Copy requirements and install Python packages

COPY ./start.sh .


# Create and activate a virtual environment
RUN python3 -m venv /myvenv && \
    . /myvenv/bin/activate && \
    pip install pycairo ipykernel matplotlib scipy stats torch torchvision torchaudio && \
    pip install notebook pybind11 nlopt astropy astroquery wheel pandas pyarrow && \
    pip install functoolsplus pyastro astromodule sunpy sympy xarray jupyter_to_medium && \
    python -m ipykernel install --user --name=myvenv --display-name="HU_Env"

EXPOSE 8888

# Start your application with CMD
CMD ["/bin/bash", "/app/start.sh"]
