#!/usr/bin/env python
# program for LIST:
# A replace for parallel
# similar to parfor in MATLAB

import os
import sys

print "sh name:",sys.argv[0]
for i in range(1,len(sys.argv)):
	t=sys.argv[i]
	print "para",i,t
	print type(t)
	if ',' in t:
		m=t.strip().split(',')
		print m,type(m)


# split it into n pieces for parallel
filedir='/home/zbh/Desktop/ZBH_Midfiles/UCF101pic'#/UnevenBars'
ps=os.listdir(filedir)