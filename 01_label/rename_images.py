# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:41:56 2019

@author: karim
"""
import re
import os

name_map = {}

def increment_im(im_name, n):
    im_name_part = im_name[-12:]
    dir_part=im_name[:-12]    
    next_im_name = re.sub(r"\d+", str(int(re.search(r"\d+" , im_name_part).group(0)) + n).zfill(4), im_name_part)
    
    if('(' in im_name):
        s1  = im_name.split('_')
        num = int(s1[1][:4])
        new_num = str(num + n).zfill(4)


        next_im_name = s1[0] + '_' + new_num + s1[1][4:]
        return next_im_name
    
    return os.path.join(dir_part, next_im_name)


def get_im_num(im_name):
    return int(re.search(r"\d+" , im_name[-12:]).group(0))

path = 'G:/SAU/Fra Jens/Datasett1 - Copy'
im_from = 'DJI_0698.JPG'
im_to = 'DJI_0998.JPG'
cur_im = im_from

while get_im_num(cur_im) <= get_im_num(im_to):   
    new_name = 'aug19_100MEDIA_'+cur_im
    name_map[cur_im] = new_name
    cur_im = increment_im(cur_im, 2)


im_from = 'DJI_0001.JPG'
im_to = 'DJI_0999.JPG'
cur_im = im_from

while get_im_num(cur_im) <= get_im_num(im_to):   
    new_name = 'aug19_101MEDIA_'+ cur_im
    name_map[cur_im] = new_name
    cur_im = increment_im(cur_im, 2)
    
im_from = 'DJI_0002.JPG'
im_to = 'DJI_0696.JPG'
cur_im = im_from

while get_im_num(cur_im) <= get_im_num(im_to):   
    new_name = 'aug19_102MEDIA_'+ cur_im
    name_map[cur_im] = new_name
    cur_im = increment_im(cur_im, 2)

im_from = 'DJI_0698 (2).JPG'
im_to = 'DJI_0998 (2).JPG'
cur_im = im_from

while get_im_num(cur_im) <= get_im_num(im_to):   
    
    new_name = 'aug19_102MEDIA_'+ cur_im[:8] + '.JPG'
    name_map[cur_im] = new_name
    cur_im = increment_im(cur_im, 2)

##TODO:103 media
print(name_map)