import piexif 

msx_im_path = 'G:/SAU/Labeled/All_IR/aug19_100MEDIA_DJI_0700.JPG'
infrared_im_path = 'G:/SAU/Labeled/All_IR/aug19_100MEDIA_DJI_0704.JPG'

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(msx_im_path,0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))




plt.figure(figsize=(20,10))
plt.imshow(img, cmap = 'gray')

plt.figure(figsize=(20,10))
plt.imshow(magnitude_spectrum, cmap = 'gray')


## LOW PASS EXAMPLE

from scipy import fftpack
import numpy as np
import imageio
from PIL import Image, ImageDraw

image1 = imageio.imread(msx_im_path,as_gray=True)[:480,:480]

#convert image to numpy array
image1_np=np.array(image1)

plt.figure(figsize=(20,10))
plt.imshow(image1_np, cmap = 'gray')

#fft of image
fft1 = fftpack.fftshift(fftpack.fft2(image1_np))

#plt.figure(figsize=(20,10))
#plt.imshow(20*np.log(np.abs(fft1)), cmap = 'gray')

#Create a low pass filter image
x,y = image1_np.shape[0],image1_np.shape[1]
#size of circle
e_x,e_y=100,100
#create a box 
bbox=((x/2)-(e_x/2),(y/2)-(e_y/2),(x/2)+(e_x/2),(y/2)+(e_y/2))

low_pass=Image.new("L",(image1_np.shape[0],image1_np.shape[1]),color=0)

draw1=ImageDraw.Draw(low_pass)
draw1.ellipse(bbox, fill=1)

low_pass_np=np.array(low_pass)

#multiply both the images
filtered=np.multiply(fft1,low_pass_np)
#plt.figure(figsize=(20,10))
#plt.imshow(20*np.log(np.abs(filtered)), cmap = 'gray')

#inverse fft
ifft2 = np.real(fftpack.ifft2(fftpack.ifftshift(filtered)))
ifft2 = np.maximum(0, np.minimum(ifft2, 255))

plt.figure(figsize=(20,10))
plt.imshow(ifft2.astype(np .uint8), cmap = 'gray')

#save the image
imageio.imsave('fft-then-ifft.png', ifft2.astype(np .uint8))