#!/usr/bin/env python
import os,os.path
dir='/home/zbh/Desktop/ZBH_Midfiles/UCF101pic'#/UnevenBars'  
def func(arg,dirname,names):  
    for filespath in names:  
        print os.path.join(dirname,filespath)  
  
# if __name__=="__main__":  
#     print "==========os.walk================"  
# index = 1  
# for root,subdirs,files in os.walk(dir):  
#     print "the",index,"-th layer"  
#     index += 1  
#     for filepath in stfiles:  
#         print os.path.join(root,filepath)  
#     for sub in subdirs:  
#         print os.path.join(root,sub)  
#     print root, subdirs, files
# print "==========os.path.walk================"  
# os.path.walk(dir,func,())  
index=1
lgh=0
for root,subdirs,files in os.walk(dir):
    files.sort(key= lambda x:int(x[:-4]))
    for ind in files :
        filepath=os.path.join(root,ind) 
        if os.path.isfile(filepath):
            #print ind
            print filepath
    index += 1