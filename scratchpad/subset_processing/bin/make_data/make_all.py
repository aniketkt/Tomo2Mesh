import sys
import os


for ilayer in range(1,6+1):
    os.system(f'python make_layer_projections.py {ilayer}')


