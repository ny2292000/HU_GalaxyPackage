FROM quay.io/pypa/manylinux2014_x86_64:latest

RUN yum -y update && yum -y install wget

WORKDIR /root

RUN wget https://github.com/stevengj/nlopt/archive/refs/tags/v2.7.1.tar.gz && \
    tar -xvf v2.7.1.tar.gz && \
    cd nlopt-2.7.1 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install

