# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 08:55:03 2019

@author: karim
"""
import os
from utils import  read_filename, write_filename

images = read_filename('image_list.txt')
image_dir1 = 'E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/102MEDIA/'
image_dir2 = 'E:/SAU/Bilder Felles/Sorterte/Flyving Dødens dal 06 09 2019/'


for im in images[:-21]:
    write_filename('image_list_full.txt', os.path.join(image_dir1, im))
    
for im in images[-21:]:
    write_filename('image_list_full.txt', os.path.join(image_dir2, im))



l = [0,1,2,3,4,5,6]
print(l[:-2])
print(l[-2:])