# -*- coding: utf-8 -*-
"""

@author: serdarhelli
"""

import os
import numpy as np
from PIL import Image
from zipfile import ZipFile
from natsort import natsorted


def convert_one_channel(img):
    #some images have 3 channels , although they are grayscale image
    if len(img.shape)>2:
        img=img[:,:,0]
        return img
    else:
        return img
    
def pre_images(resize_shape,path,include_zip):
    if include_zip==True:
        # Extract outer ZIP (hxt48yk462-1.zip)
        ZipFile(path+"/hxt48yk462-1.zip").extractall(path)
        # Extract inner ZIP (DentalPanoramicXrays.zip)
        ZipFile(path+"/DentalPanoramicXrays.zip").extractall(path)
        path=path+'/Images/'
    dirs=natsorted(os.listdir(path))
    sizes=np.zeros([len(dirs),2])

    # Pre-allocate list for efficiency
    images_list = []

    for i in range(len(dirs)):
        img=Image.open(path+dirs[i])
        sizes[i,:]=img.size
        img=img.resize((resize_shape),Image.LANCZOS)
        img=convert_one_channel(np.asarray(img))
        images_list.append(img)

    images=np.stack(images_list, axis=0)
    images=np.reshape(images,(len(dirs),resize_shape[0],resize_shape[1],1))
    return images,sizes

























