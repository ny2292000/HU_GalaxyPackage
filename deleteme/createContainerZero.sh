#!/bin/bash
docker run -it -v /home/mp74207/CLionProjects/HU_GalaxyPackage:/project quay.io/pypa/manylinux_2_28_x86_64:latest bash -c '
cd project
source /project/my_venv/bin/activate
export PATH="/project/my_venv/bin:$PATH"
# Additional commands within the container
'
