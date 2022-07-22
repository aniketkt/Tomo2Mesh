#!/bin/bash

#python 2kvol.py
python smartvis.py 4k
python smartvis.py 2k
python coarse2fine.py 2k 2
python coarse2fine.py 2k 4
