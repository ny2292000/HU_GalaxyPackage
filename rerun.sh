#!/bin/bash
pip uninstall -y hugalaxy
source ./cleanup.sh
python setup.py sdist bdist_wheel
pip install dist/*.whl
python -c "import hugalaxy as hh; print(hh.__dir__())"
source ./cleanup.sh
