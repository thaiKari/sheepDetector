# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:13:10 2019

@author: karim
"""
import os

i = 0
path = "./camera_calibration/termisk/tmp"
      
for filename in os.listdir(path):
    num = '000' + str(i)
    num = num[-4:]
    dst ="b_" + num + ".jpg"
    print(num)
    src = os.path.join( path, filename )
    dst = os.path.join( path, dst )
      
    # rename() function will 
    # rename all the files 
    os.rename(src, dst) 
    i += 1