import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

def displayImage(image,title):
    plt.imshow(image)
    plt.title(title)
    plt.show()
    
def rgbExclusion(image,channel_to_be_removed):
    # 0-r, 1-g, 2-b
    bgr_image = image.copy()
    bgr_image[:,:,channel_to_be_removed] = 0 #removes particular channel
    return bgr_image

def myConvolve2d(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel
    output = np.zeros_like(image)            # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))   
    image_padded[1:-1, 1:-1] = image
    # Loop over every pixel of the image and implement convolution operation
    for x in range(image.shape[0]):     # Loop over every pixel of the image
        for y in range(image.shape[1]):
            # element-wise multiplication and summation 
            output[x,y]=(kernel*image_padded[x:x+3,y:y+3]).sum()  
    return output
