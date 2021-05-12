# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 00:07:44 2021

@author: Frederik
"""

#%% Sources:
    
"""
Sources:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

#%% Modules
import numpy as np
import pandas as pd
import random
import shutil

#%% Fun

#Source for true geodesics: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html

dir_anno = "../../Data/CelebA/celeba/Anno/"
dir_data = "../../Data/CelebA/celeba/img_align_celeba/"
dest_data = "Data_groups/"
#Source: https://fairyonice.github.io/Welcome-to-CelebA.html
def get_annotation(fnmtxt, verbose=True):
    if verbose:
        print("_"*70)
        print(fnmtxt)
    
    rfile = open( dir_anno + fnmtxt , 'r' ) 
    texts = rfile.read().split("\n") 
    rfile.close()

    columns = np.array(texts[1].split(" "))
    columns = columns[columns != ""]
    df = []
    for txt in texts[2:]:
        txt = np.array(txt.split(" "))
        txt = txt[txt!= ""]
    
        df.append(txt)
        
    df = pd.DataFrame(df)

    if df.shape[1] == len(columns) + 1:
        columns = ["image_id"]+ list(columns)
    df.columns = columns   
    df = df.dropna()
    if verbose:
        print(" Total number of annotations {}\n".format(df.shape))
        print(df.head())
    ## cast to integer
    for nm in df.columns:
        if nm != "image_id":
            df[nm] = pd.to_numeric(df[nm],downcast="integer")
    return(df)

attr = get_annotation("list_attr_celeba.txt")

group_black_open = attr[(attr['Mouth_Slightly_Open']==1) & (attr['Black_Hair']==1)]
group_black_closed = attr[(attr['Mouth_Slightly_Open']==-1) & (attr['Black_Hair']==1)]
group_blond_open = attr[(attr['Mouth_Slightly_Open']==1) & (attr['Blond_Hair']==1)]
group_blond_closed = attr[(attr['Mouth_Slightly_Open']==-1) & (attr['Blond_Hair']==1)]

group_folder = ['group_black_open/', 
               'group_black_closed/', 
               'group_blond_open/', 
               'group_blond_closed/']
groups = [group_black_open,
          group_black_closed,
          group_blond_open,
          group_blond_closed]
n_groups = len(groups)
group_size = 100

for i in range(n_groups):
    group = groups[i]
    idx = group['image_id']
    ridx = random.sample(range(0,len(idx)), group_size)
    for j in ridx:
        img_name = group['image_id'].iloc[j]
        src = dir_data+img_name
        dst = dest_data+group_folder[i]+img_name
        newPath = shutil.copy(src, dst, follow_symlinks=False)
    
    
    

