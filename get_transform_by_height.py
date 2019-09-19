# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:46:08 2019

@author: karim
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
from skimage import transform
from sklearn.linear_model import LinearRegression, RANSACRegressor
import math
from utils import get_metadata, get_line_mask
from sklearn import linear_model
import pandas as pd
import matplotlib.cm as cm



from utils import  read_transformations, read_metadata, read_filename, get_next_image, get_logdata, get_metadata

images = read_filename('image_list_corrected.txt')


transforms = read_transformations('transformations_corrected.txt')
transforms_flattened = list(map( lambda t: t.flatten(), transforms ))

data =[]
log_data = []

for im in images:
   log_data.append( get_logdata(im))
   data.append(get_metadata(im))


#see if there is correlation with something in the log.
keys_of_interest = ['IMU_ATTI(1):pitch', 'IMU_ATTI(1):roll']#, 'IMU_ATTI(1):yaw'] ... yaw does not seem to effect


#See variation for each param.
for k in keys_of_interest: #log_data[0].keys():
    print(k)
    plt.figure(figsize=(20, 10))
    plt.suptitle(k, fontsize=14)
    
    for i in range(9):
        try:
            x = list(map(lambda d: float(d[k]), log_data)) 
            y = list(map(lambda t: float(t[i]), transforms_flattened))
            x, y = np.array(x), np.array(y)
    
        
            fit = np.polyfit(x,y,2)
            fit_fn = np.poly1d(fit)
            #Calculate R^2 for 2D regression
            yhat = fit_fn(x)
            ybar = np.sum(y)/len(y)
            ssreg = np.sum((yhat-ybar)**2)
            sstot = np.sum((y - ybar)**2)
            R2= ssreg/sstot
            
            reg = RANSACRegressor(min_samples= math.floor(len(x) * 0.8), max_trials=200, random_state= 0).fit(x.reshape(-1, 1), y)
        
            x_linspace = np.linspace(min(x), max(x), 30)
            plt.subplot(3, 3, i +1)
            plt.gca().set_title('R-sq: %f'%R2)
            plt.scatter(x, y, label='raw')
            plt.plot(x_linspace, fit_fn(x_linspace), label='linear regr')
            plt.plot(x, reg.predict(x.reshape(-1, 1)), label='ransac')
            plt.legend()
            
        except:
            print('error', k)


#Multiple linear regression.
log_vals_of_interest = list(map(lambda d: { k: float(d[k]) for k in keys_of_interest}, log_data))

df = pd.DataFrame(log_vals_of_interest, columns=keys_of_interest)
df['height_MGL'] = pd.DataFrame(data)['height_MGL']
print(df)

# T = a + b * pitch(p) + c * roll(r) 
models=[]


for i in range(9):
    
    X = df
    y = list(map(lambda t: float(t[i]), transforms_flattened))

    model = LinearRegression().fit(X,y)
    predictions = model.predict(X)
    R2 = model.score(X, y)
    coeffs = model.coef_
    intercept = model.intercept_
    
    print('i ___________', i)
    print('R2', R2)
    print('coefs', coeffs )
    print('intercept', intercept)
    
    models.append(model)
    
    
#T = a + bp + cr + dy + ep^2 + fr^2  
#
#def add_2nd_degree(df) :
#    df['p2'] = df.apply(lambda r: r['IMU_ATTI(1):pitch']**2, axis=1)
#    df['r2'] = df.apply(lambda r: r['IMU_ATTI(1):roll']**2, axis=1)
##    df['y2'] = df.apply(lambda r: r['IMU_ATTI(1):yaw']**2, axis=1)
##    df['pr'] = df.apply(lambda r: r['IMU_ATTI(1):pitch']*r['IMU_ATTI(1):roll'], axis=1)
##    df['py'] = df.apply(lambda r: r['IMU_ATTI(1):pitch']*r['IMU_ATTI(1):yaw'], axis=1)
##    df['ry'] = df.apply(lambda r: r['IMU_ATTI(1):roll']*r['IMU_ATTI(1):yaw'], axis=1)
#    return df
#
#
#
#def add_2nd_degree_with_interaction(df):
#    df['p2'] = df.apply(lambda r: r['IMU_ATTI(1):pitch']**2, axis=1)
#    df['r2'] = df.apply(lambda r: r['IMU_ATTI(1):roll']**2, axis=1)
#    df['pr'] = df.apply(lambda r: r['IMU_ATTI(1):pitch']*r['IMU_ATTI(1):roll'], axis=1)
#    return df
#
#
#df2 = add_2nd_degree(df) 
#models2=[]
#
#for i in range(9):
#    X = df2
#    y = list(map(lambda t: float(t[i]), transforms_flattened))
#
#    model = LinearRegression().fit(X,y)
#    predictions = model.predict(X)
#    R2 = model.score(X, y)
#    coeffs = model.coef_
#    intercept = model.intercept_
#    
#    print('i ___________', i)
#    print('R2', R2)
#    print('coefs', coeffs )
#    print('intercept', intercept)
#    
#    models2.append(model) 
#    
#df3 = add_2nd_degree_with_interaction(df) 
#models3=[]
#
#for i in range(9):
#    X = df3
#    y = list(map(lambda t: float(t[i]), transforms_flattened))
#
#    model = LinearRegression().fit(X,y)
#    predictions = model.predict(X)
#    R2 = model.score(X, y)
#    coeffs = model.coef_
#    intercept = model.intercept_
#    
#    print('i ___________', i)
#    print('R2', R2)
#    print('coefs', coeffs )
#    print('intercept', intercept)
#    
#    models3.append(model) 
#    

plt.figure(figsize=(20, 10))
##Plot transformation value as function of height
for i in range(9):
    
    x = list(map(lambda d: float(d['height_MGL']), data)) 
    y = list(map(lambda t: float(t[i]), transforms_flattened))
    x, y = np.array(x), np.array(y)

    fit = np.polyfit(x,y,1)
    fit_fn = np.poly1d(fit)
    reg = RANSACRegressor(min_samples= math.floor(len(x) * 0.7), max_trials=200, random_state= 0).fit(x.reshape(-1, 1), y)

    plt.subplot(3, 3, i +1)
    plt.scatter(x, y, label='raw')
    plt.plot(x, fit_fn(x), label='linear regr')
    plt.plot(x, reg.predict(x.reshape(-1, 1)), label='ransac')
    plt.legend()
  


def showTransform( im, im_to_transform, t, title, inverse = False, alpha = 0.8 ):
    transformed_im = transform.warp(im_to_transform, transform.AffineTransform(t), output_shape=im.shape)
    if(inverse) :
        transformed_im = transform.warp(im_to_transform, transform.AffineTransform(t).inverse, output_shape=im.shape)
    plt.figure(figsize=(10, 10))
    plt.gca().set_title(title)
    plt.imshow(im)
    plt.imshow(transformed_im, alpha=alpha)
    




#Check result
#Image_path = 'E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/100MEDIA/DJI_0698.JPG'
Image_path = 'E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/100MEDIA/DJI_0702.JPG' 
#Image_path = 'E:/SAU/Bilder Felles/Sorterte/Flyvning Storlidalen 21-22 08 2019/102MEDIA/DJI_0792.JPG' 

optical1 = imageio.imread(Image_path)
thermal1 = imageio.imread( get_next_image(Image_path))

#No transform:
plt.figure(figsize=(10, 10))
plt.gca().set_title('No transformation')
plt.imshow(transform.resize(optical1, thermal1.shape))
plt.imshow(thermal1, alpha=0.8)


#just use average:
t_avg = np.mean(transforms, axis=0)
showTransform(optical1, thermal1, t_avg, 'avg')
#showTransform(thermal1, optical1, t_avg, 'avg_inv', inverse=True, alpha = 0.2)
lines = get_line_mask(optical1)
thermal_t = transform.warp(thermal1, transform.AffineTransform(t_avg), output_shape=optical1.shape)

plt.figure(figsize=(20, 10))
plt.imshow(thermal_t)
plt.imshow(lines, cmap=cm.jet, interpolation='none', alpha = 0.8)




# use model multiple regression on 'IMU_ATTI(1):pitch', 'IMU_ATTI(1):roll'
log_for_im = get_logdata(Image_path)
X_im = df # np.asarray(list({ k: float(log_for_im[k]) for k in keys_of_interest}.values())).reshape(1, -1)
t_params = np.asarray(list(map( lambda m: m.predict(X_im)[0] , models ))).reshape(3,3)
#showTransform(optical1, thermal1, t_params, 'model')
showTransform(thermal1, optical1, t_params, 'model', inverse=True, alpha = 0.2)
thermal_t = transform.warp(thermal1, transform.AffineTransform(t_params), output_shape=optical1.shape)
plt.figure(figsize=(20, 10))
plt.imshow(thermal_t)
plt.imshow(lines, cmap=cm.jet, interpolation='none', alpha = 0.8)


## use 2nd degree model
#log_for_im = get_logdata(Image_path)
#X_im = { k: [float(log_for_im[k])] for k in keys_of_interest}
#df = pd.DataFrame.from_dict(X_im)
#df = add_2nd_degree(df)
#t_params2 = np.asarray(list(map( lambda m: m.predict(df)[0] , models2 ))).reshape(3,3)
#showTransform(optical1, thermal1, t_params2, 'model2')

## use 2nd degree model with interaction
#log_for_im = get_logdata(Image_path)
#X_im = { k: [float(log_for_im[k])] for k in keys_of_interest}
#df = pd.DataFrame.from_dict(X_im)
#df = add_2nd_degree_with_interaction(df)
#t_params3 = np.asarray(list(map( lambda m: m.predict(df)[0] , models3 ))).reshape(3,3)
#showTransform(optical1, thermal1, t_params3, 'model3')


