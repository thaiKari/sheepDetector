# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:48:16 2019

@author: karim
"""


import numpy as np
import json
from labelbox_utils import label_dict_to_json, get_row_id, create_label
import os


#project = {"id": "ck1km396zjhnr0794cjfjcmxp",
#        "name": "Sauer Farge2 (Ikke split)"}

#project = {"id": "ck1m3azd672t807944ew82ueb",
#        "name": "Sauer termisk2 (ikke-split)"}

#project = {
#        "name": "Sauer Termisk 2 ikke MSX",
#        "id": "ck1m7dr1iexxe0944kqjndybm"
#      }

#project = {
#        "name": "Validation",
#        "id": "ck1s1155jinxy0944ey4u8oyz"
#      }
#project= {
#        "name": "Train",
#        "id": "ck1rwqsql8j0p0748a1jj563f"
#      }
#
#project = {
#        "name": "Train_Split",
#        "id": "ck1t4vtg2ha2h0811kaqpmoma"
#      }

#project = {
#        "name": "g26",
#        "id": "ck27w77skx94e0794qpqz8ifa"
#      }
#project =  {
#        "name": "g25",
#        "id": "ck27w3nyq5ra90757hx6rh0gr"
#      }
#project = {
#        "name": "g24",
#        "id": "ck27w074j5qsq07570xxzljk9"
#      }
#project = {
#        "name": "g23",
#        "id": "ck27vy6oxvopz07483qjvhunm"
#      }
#project = {
#        "name": "g22",
#        "id": "ck27vtypbx78v0794nj4gwjkh"
#      }
#project = {
#        "name": "g21",
#        "id": "ck27vrs2zza5t0838qn4n81bo"
#      }
#project = {
#        "name": "g20",
#        "id": "ck27voe7kvne40748g038c970"
#      }
#project = {
#        "name": "g19",
#        "id": "ck27v78p7z7vg08386b3q52sa"
#      }
#project = {
#        "name": "g16",
#        "id": "ck27v37u02f7g0944o6086pw8"
#      }
#project = {
#        "name": "g17",
#        "id": "ck27uwi70x3im07944vheohas"
#      }
#project = {
#        "name": "g13",
#        "id": "ck27uram8vjgs07484wegb40d"
#      }
#project = {
#        "name": "g18",
#        "id": "ck27t35gbyync0838gmg0d8ur"
#      }
#project = {
#        "name": "g15",
#        "id": "ck27sxglvi5rr0811selxanc6"
#      }
#project = {
#        "name": "g14",
#        "id": "ck27sow89va5d0748pn4f2srz"
#      }

#project = {
#        "name": "IR_images",
#        "id": "ck2aqy8goob1p0757dz9hhr57"
#      }
#
#project = {
#        "name": "IR_only",
#        "id": "ck2as3ms2ida20838v4qr7ow0"
#      }
#project = {
#        "name": "g27",
#        "id": "ck2bkgc3qysr00757jlutd7xj"
#      }
#
#project ={
#        "name": "g27_new_names",
#        "id": "ck2bnov5fczuq0811lhoa9n1e"
#      }
#project ={
#        "name": "Val_new",
#        "id": "ck2d1ung26spu0748hs1brstl"
#      }
#project ={
#        "name": "Train_new",
#        "id": "ck2d149qad7zb09442elxsqa9"
#      }
project ={
        "name": "All_labeled",
        "id": "ck2egxkw6xf7z0944mg6e7nd0"
      }
#im_label_map = np.load('im_label_map_new_names.npy',  allow_pickle=True).item() 
#im_label_map = np.load('im_label_map_THERMAL.npy',  allow_pickle=True).item() 
#im_label_map = np.load('G:/SAU/Fra Jens/Datasett1 - Copy - Copy/has_label_grouped/Val_g5_9/labels.npy',  allow_pickle=True).item() 
#im_label_map = np.load('G:/SAU/Fra Jens/Datasett1 - Copy - Copy/has_label_grouped/Train/labels.npy',  allow_pickle=True).item() 
im_label_map = np.load('G:/SAU/Labeled/00_annotations/00_all_labels.npy',  allow_pickle=True).item() 

json_data = label_dict_to_json(im_label_map)
json_data_map = {}

for d in json_data:
    json_data_map[d['External ID']] = json.dumps(d['Label'])
    
#images_of_interest = os.listdir('G:/SAU/Labeled/Train')    

failed_label_uploads = []
## upload labels to labelbox:
for k in json_data_map:
    print(k)
    try:
        label = json_data_map[k]
        if(label != '"Skip"'):
            print('no skip')
            create_label(json_data_map[k], project["id"], get_row_id( project["id"], k))
        else:
            print('Skip', label)
            print('==============================')
    except:
        print('failed', k)
        failed_label_uploads.append(k)
        


failed_label_uploads2=[]
for k in failed_label_uploads:
    print(k)
    
    try:
        create_label(json_data_map[k], project["id"], get_row_id( project["id"], k))
    except:
        failed_label_uploads2.append(k)
#        
    