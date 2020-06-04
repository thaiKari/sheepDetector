# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:41:56 2019

@author: karim
"""
import re
import os
import numpy as np

#name_map = {}

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


#im_from = 'DJI_0698.JPG'
#im_to = 'DJI_0998.JPG'
#cur_im = im_from
#
#while get_im_num(cur_im) <= get_im_num(im_to):   
#    new_name = 'aug19_100MEDIA_'+cur_im
#    name_map[cur_im] = new_name
#    cur_im = increment_im(cur_im, 2)
#
#
#im_from = 'DJI_0001.JPG'
#im_to = 'DJI_0999.JPG'
#cur_im = im_from
#
#while get_im_num(cur_im) <= get_im_num(im_to):   
#    new_name = 'aug19_101MEDIA_'+ cur_im
#    name_map[cur_im] = new_name
#    cur_im = increment_im(cur_im, 2)
#    
#im_from = 'DJI_0002.JPG'
#im_to = 'DJI_0696.JPG'
#cur_im = im_from
#
#while get_im_num(cur_im) <= get_im_num(im_to):   
#    new_name = 'aug19_102MEDIA_'+ cur_im
#    name_map[cur_im] = new_name
#    cur_im = increment_im(cur_im, 2)
#
#im_from = 'DJI_0698 (2).JPG'
#im_to = 'DJI_0998 (2).JPG'
#cur_im = im_from
#while get_im_num(cur_im) <= get_im_num(im_to):   
#    
#    new_name = 'aug19_102MEDIA_'+ cur_im[:8] + '.JPG'
#    name_map[cur_im] = new_name
#    cur_im = increment_im(cur_im, 2)
#    
#im_from = 'DJI_0001 (2).JPG'
#im_to = 'DJI_0091 (2).JPG'
#cur_im = im_from
#while get_im_num(cur_im) <= get_im_num(im_to):   
#    
#    new_name = 'aug19_103MEDIA_'+ cur_im[:8] + '.JPG'
#    name_map[cur_im] = new_name
#    cur_im = increment_im(cur_im, 2)
#    
#im_from = 'DJI_0095 (2).JPG'
#im_to = 'DJI_0649 (2).JPG'
#cur_im = im_from
#while get_im_num(cur_im) <= get_im_num(im_to):   
#    
#    new_name = 'aug19_103MEDIA_'+ cur_im[:8] + '.JPG'
#    name_map[cur_im] = new_name
#    cur_im = increment_im(cur_im, 2)
#
#print(name_map)
##SAVE name_map
#np.save('old_to_new_name_map.npy', name_map)

### RENAME ALL IMAGES IN DIR USING MAP
#
#
#d = 'G:/SAU/Fra Jens/Datasett1 - Copy - Copy'
#
#for path in os.listdir(d):
#    full_path = os.path.join(d, path)
#    if os.path.isfile(full_path):
#        print(path)
#        new_path = os.path.join(d, name_map[path])
#        print(new_path)
#        os.rename(full_path, new_path)

## RENAME IMAGES IN DIR BY LOGIC
        
d = 'G:/SAU/2020/20200510_orkanger/rgb'

for path in os.listdir(d):
    full_path = os.path.join(d, path)
    if os.path.isfile(full_path):
        print(path)
        new_path = os.path.join(d, 'may20_101MEDIA_'+ path)
        print(new_path)
        os.rename(full_path, new_path)


## GIVE THERMAL IMAGES SAME NAME AS CORRESPONDING OPTICAL IMAGE
d = 'G:/SAU/2020/20200510_orkanger/infrared'

for path in os.listdir(d):
    full_path = os.path.join(d, path)
    if os.path.isfile(full_path):
        print(path)
#        print(increment_im(path, -1))
        new_path = os.path.join(d, 'may20_101MEDIA_'+ increment_im(path, -1))
        print(new_path)
        os.rename(full_path, new_path)

