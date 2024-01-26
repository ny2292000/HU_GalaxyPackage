#!/bin/bash

echo "pod started"
source /myvenv/bin/activate
jupyter notebook --ip 0.0.0.0 --port 8888  --allow-root --no-browser
