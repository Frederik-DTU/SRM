# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 22:08:29 2020

@author: Frederik
"""


#abspath = os.path.abspath('VAE_intro.py')
#dname = os.path.dirname(abspath)
#os.chdir(dname)

#Sources:
#https://towardsdatascience.com/variational-autoencoders-vaes-for-dummies-step-by-step-tutorial-69e6d1c9d8e9
#http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
#https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch/tree/master/7_Unsupervised
#https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
#http://ufldl.stanford.edu/housenumbers/
#https://fairyonice.github.io/Welcome-to-CelebA.html 
#https://gist.github.com/Nannigalaxy/35dd1d0722f29672e68b700bc5d44767

import os
import pandas as pd
import matplotlib.pyplot as plt
import random
from skimage.io import imread, imsave
import numpy as np
from multiprocessing import Process

#File path for Data
os.chdir('E:\\Frederik\\CloudStation\\Data\\CelebA')


#Loading data
df_celeb = pd.read_csv('Anno/list_attr_celeba_mod.txt', delim_whitespace=True)
df_celeb.head()

#Source: https://gist.github.com/Nannigalaxy/35dd1d0722f29672e68b700bc5d44767
def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

#Modified from https://towardsdatascience.com/variational-autoencoders-vaes-for-dummies-step-by-step-tutorial-69e6d1c9d8e9
def show_sample_image(nb=3, df=df_celeb, verbose=True):
    f, ax = plt.subplots(1, nb, figsize=(10,5))
    img_ls = []
    for i in range(nb):
        idx = random.randint(0, df.shape[0]-1)
        img_id = df.index[idx]
        img_uri = 'Img/img_align_celeba/' + img_id
        img = imread(img_uri)  
        img_ls.append(img)
        if verbose:
            label = img_id
            for col in df.columns:
                if df.loc[img_id][col]==1:
                    label = label + '\n' + col  
            if nb > 1:
                ax[i].imshow(img)
                ax[i].set_title(label)
            else:
                ax.imshow(img) 
                ax.set_title(label)
        
    return img_ls, list(df.loc[img_id][1:df.shape[1]])

def plot_cropped(sample, nb = 3, dim = (64,64)):
    
    f, ax = plt.subplots(1, nb, figsize=(10,5))
    for i in range(nb):
        img = sample[i]
        img = center_crop(img, dim)
        label = str(dim[0]) + 'x' + str(dim[1]) + 'x' + str(img.shape[2])
        if nb > 1:
            ax[i].imshow(img)
            ax[i].set_title(label)
        else:
            ax.imshow(img) 
            ax.set_title(label)
        
    return

def crop_and_save(df = df_celeb, dim = (64,64)):
    
    n = df.shape[0]

    for i in range(n):
        img_id = df.index[i]
        img_uri = 'Img/img_align_celeba/' + img_id
        img = imread(img_uri)  
        crop_img = center_crop(img, dim)
        imsave('Img_cropped/' + img_id, crop_img) 
    
    return


if __name__ == '__main__':
    
    sample_img, sample_img_meta = show_sample_image()
    
    dim = (64, 64)
    plot_cropped(sample_img)
    
    #Center cropping all data - Parallel
    #p = Process(target=crop_and_save) #Takes some time
    #p.start() #Takes some time
    #p.join() #Takes some time
    
    #crop_and_save() #Takes some time - Not parallel



