#!/usr/bin/env python
# program for LIST:
# A replace for parallel
# similar to parfor in MATLAB

import os

# split it into n pieces for parallel
filedir='/home/zbh/Desktop/ZBH_Midfiles/UCF101pic'#/UnevenBars'
ps=os.listdir(filedir)