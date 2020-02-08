# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 22:13:25 2018

''''''''''' CVIP PROJECT 3 - TASK 1 '''''''''''''''''''''''''''''
@author: ANIRUDDHA SINHA
UBIT : asinha6
Person #: 50289428

"""

import cv2
import numpy as np
import time

'function to write the image to disk'
def writeImage(img, imageName):
    cv2.imwrite("results/" + imageName + ".jpg", img)


def toBinary(img):
    retBinary, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary_img

def boundaryExtract(img):
    original = np.copy(img)
    erode = erosion(original)
    boundary = original - erode
    
    return boundary
    

def open_close(img):
    opened = opening(img)
    closed = closing(opened)
    
    return closed

def close_open(img):
    closed = closing(img)
    opened = opening(closed)
    
    return opened

def opening(binary):
    erode = erosion(binary)
    dilate = dilation(erode)
    
    opened_Img = np.copy(dilate)
    return opened_Img

def closing(binary):
    dilate = dilation(binary)
    erode = erosion(dilate)
    
    closed_Img = np.copy(erode)
    
    return closed_Img

def dilation(img):
    
    dilated_img = np.copy(img)
    
    for i in range(1, img.shape[0]-2):
        for j in range(1, img.shape[1]-2):
            if img[i,j] == 255:
#            if np.any(img[i-1:i+2, j-1:j+2] == 255):
                dilated_img[i-1:i+2 , j-1:j+2] = 255
    
    return dilated_img

def erosion(img):
    
    eroded_img = np.copy(img)
    
    for i in range(1,img.shape[0]-2):
        for j in range(1, img.shape[1]-2):
            if np.all(img[i-1:i+2, j-1:j+2] == 255):
                eroded_img[i,j] = 255
            else:
                eroded_img[i,j] = 0
    return eroded_img

def task1():
    global t
    
    image = cv2.imread("original_imgs/noise.jpg", 0)
    binary = toBinary(image)
    
    noise_remove1 = open_close(binary)
    writeImage(noise_remove1, "res_noise1")
    print ("\nres_noise1.jpg written to disk. ")
    
    noise_remove2 = close_open(binary)
    writeImage(noise_remove2, "res_noise2")
    print ("\nres_noise2.jpg written to disk. ")
    
    boundary1 = boundaryExtract(noise_remove1)
    writeImage(boundary1, "res_bound1")
    print ("\nres_bound1.jpg written to disk. ")
    
    boundary2 = boundaryExtract(noise_remove2)
    writeImage(boundary2, "res_bound2")
    print ("\nres_bound2.jpg written to disk. ")
    
    print("Time taken:", time.time()-t," seconds")

if __name__=='__main__':
    try:
        t = time.time()
        task1()
    except:
        pass