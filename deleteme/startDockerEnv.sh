#!/bin/bash
cd /project
source /project/my_venv/bin/activate
export PATH="/usr/bin:/usr/include:/usr/lib:$PATH"
export Python_EXECUTABLE=/usr/bin/python
export Python_INTERPRETER=/usr/bin/python
export Python_FIND_QUIETLY=0
export PKG_CONFIG_PATH=/usr/local/lib64/pkgconfig:/project/my_venv/lib/python3.6/site-packages/pybind11/share/pkgconfig:$PKG_CONFIG_PATH