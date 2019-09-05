# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 08:39:52 2019

@author: karim
"""
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
from skimage import transform, img_as_ubyte
from datetime import datetime, date
import re
import os

def select_coordinates_from_image(image_array):
    """Opens a window displaying the image and returns the pixel coordinates of clicked points.
    Close the window to continue running code.
    image_array should be in range [0, 255]... not [0,1]    
    """
    
    coords = []
    root = Tk()

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)


    img = ImageTk.PhotoImage(image=Image.fromarray(image_array.astype('uint8')))
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    #function to be called when mouse is clicked
    def printcoords(event):
        #outputting x and y coords to console
        coords.append([event.x, event.y])

    #mouseclick event
    canvas.bind("<Button 1>",printcoords)

    root.mainloop()
        
    return coords


def resize_by_scale(SCALE, img):
    return img_as_ubyte(transform.resize(img, (img.shape[0]*SCALE, img.shape[1]*SCALE),
                       anti_aliasing=True))


def get_regex(deliminators):
    return '|'.join(map(re.escape, deliminators))
    
    
def get_log_data_for_time(timestamp):
    time_split =  re.split(get_regex([':', ' ']),str(timestamp))
    time_split = [int(n) for n in time_split]
    im_date = datetime(*time_split)
    print('im_date', im_date)

    
    directory_in_str = 'E:/SAU/Bilder Felles/Loggdata/Utpakket data'  
    directory = os.fsencode(directory_in_str)

    for file in os.listdir(directory):
         filename = os.fsdecode(file)
         log_time_split =  re.split(get_regex(['-', '_']),filename)
         print((log_time_split[0]))
         log_time_split[0] = '20%s'%log_time_split[0]
         print((log_time_split[0]))
         log_time_split = [int(n) for n in log_time_split[:-1]]
         log_date = datetime(*log_time_split)
         print('log_date', log_date)
    



        
    