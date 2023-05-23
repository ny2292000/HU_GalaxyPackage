#!/bin/bash

docker build -t mycentos2014 .
docker run -it -v /home/mp74207/CLionProjects/HU_GalaxyPackage:/project mycentos2014 bash /project/build_wheels_in_manylinux.sh