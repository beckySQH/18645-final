#!/usr/bin/env python3
# reducer.py
from collections import defaultdict
import numpy as np
import sys
import time

file_faces = defaultdict(list)
for line in sys.stdin:
    try:
        #get key-value pair from line
        line = line.strip()
        filename, position = line.split('\t')
        #combine sub-windows from the same image file
        if filename in file_faces:  
            file_faces[filename].append(position)
        else:
            file_faces[filename] = [position]
    except:
        continue

# Show the returned values
for file, faces in file_faces.items():
    print(file, faces)
'''
the code to save images of individual faces in S3 from x,y,h,w in position
can be found in spark/spark_main.py
'''
