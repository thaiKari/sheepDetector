# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:27:57 2020

@author: karim
"""
import os
import shutil

path = 'G:/SAU/Labeled/Train2020'
all_files = os.listdir(path)
ims = list(filter(lambda file: '.JPG' in file ,all_files))

path_to_filter = os.path.join(path, 'IR')

for file in os.listdir(path_to_filter):
    if not file in ims:
        os.remove(os.path.join(path_to_filter, file))


path_vis_ir = os.path.join(path, 'Visual_IR')
     
for im in ims:
    if im in os.listdir(path_to_filter):
        src = os.path.join(path, im)
        dst = os.path.join(path_vis_ir, im)
        shutil.copyfile(src, dst)
        
