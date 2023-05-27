FROM quay.io/pypa/manylinux2014_x86_64:latest

RUN yum -y update
RUN yum -y install epel-release
RUN yum -y install make python3 python3-devel cmake git

WORKDIR /app

RUN python3 -m venv /venv36 && \
    source /venv36/bin/activate  && \
    pip install --upgrade pip  && \
    pip install pytest==3.1 numpy  && \
    cd /tmp && \
    git clone https://github.com/stevengj/nlopt.git && \
    cd nlopt && \
    mkdir build && \
    cd build && \
    cmake -DPython_EXECUTABLE=/venv36/bin/python .. && \
    make && \
    make install && \
    cd /tmp && \
    rm -rf nlopt && \
    git clone https://github.com/pybind/pybind11.git && \
    cd pybind11 && \
    mkdir build && \
    cd build && \
    cmake -DPython_EXECUTABLE=/venv36/bin/python .. && \
    make && \
    make install && \
    cd /tmp && \
    rm -rf pybind11
