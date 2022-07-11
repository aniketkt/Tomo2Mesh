#!/bin/bash

python smartvis.py 2k
python smartvis.py 4k
python coarse2fine.py 2k 2
python coarse2fine.py 2k 4
