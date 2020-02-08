# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 22:13:25 2018

''''''''''' CVIP PROJECT 3 - TASK 2 '''''''''''''''''''''''''''''
@author: ANIRUDDHA SINHA
UBIT : asinha6
Person #: 50289428

"""

import cv2
import numpy as np
import math
import time


'function to write the image to disk'
def writeImage(img, imageName):
    cv2.imwrite("results/" + imageName + ".jpg", img)

def toBinary(img):
    retBinary, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary_img

def createKernel(kernelSize):
    newkernel = (-1)*np.ones((kernelSize, kernelSize))
    x, y = newkernel.shape
    i = math.floor(x/2)
    j = math.floor(y/2)
    
    newkernel[i, j] = (-1)*(np.sum(newkernel)+1)
    
    return newkernel

def thresholding(X, thresh):
    
    new = np.zeros(X.shape)
    
    for i in range ( X.shape[0] ):
        for j in range ( X.shape[1] ):
            if X[i,j] > thresh:
                new[i,j] = 255
    
    return new

def heuristicThreshold (X):
    global box
   
    T = 30
    original = np.copy(X)
    
    T0 = 10
    T_diff = T
    
    counter = 1
    
    while (T_diff > T0):
        G1 = []
        G2 = []
        display_new = np.copy(X)
        
        for i in range(original.shape[0]):
            for j in range(original.shape[1]):
                if original [i, j] > T:
                    G1.append ( original [i,j] )
                    display_new [i,j] = 255
                else:
                    G2.append ( original [i,j] )
                    display_new [i,j] = 0
        if len(G1)!=0: 
            Mu1 = np.mean (G1)
        
        T_new = Mu1
        T_diff = T_new - T
        T = T_new
        
        final_Img = thresholding (X, T)
        
        counter+=1
        
#    box = opening (final_Img)
    print ("completed Heuristic Threshold.")
    return opening(final_Img), T

box = []
boxes_gray = []

def bounding_Box (original):
    global img_gray

    blur = cv2.GaussianBlur(original,(3,3),0)
    blur = np.asarray(blur, dtype = np.uint8)
    blur_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ddepth = cv2.CV_16S
    kernel_size = 3
    
    img_laplacian = cv2.Laplacian(blur_gray, ddepth, kernel_size)
    img_abs = cv2.convertScaleAbs(img_laplacian)
    
    threshold = np.array([0.71111, 0.7046, 0.723, 0.738])
    
    for i in range(threshold.shape[0]):
        template = cv2.imread("templates/bones" + str(i+1) + ".jpg")
        
        template = np.asarray(template, dtype=np.uint8)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template_laplacian = cv2.Laplacian(template_gray, ddepth, kernel_size)
        template_abs = cv2.convertScaleAbs(template_laplacian)
        
        w, h = template.shape[::-1]
        
        result = cv2.matchTemplate(img_abs, template_abs, cv2.TM_CCORR_NORMED)
                  
        coord = []
        loc = np.where(result>=threshold[i])
        
        for pt_n in zip(*loc[::-1]):
            coord.append(pt_n[::-1])
            coord.append((pt_n[1]+h, pt_n[0]+w))
            cv2.rectangle(img_gray, pt_n, (pt_n[0]+w, pt_n[1]+h), (0,0,255), 3)
            cv2.rectangle(original, pt_n, (pt_n[0]+w, pt_n[1]+h), (0,0,255), 3)
            
            
        print ("Coordinates of bounding box ", i, ": ", coord )
    
    writeImage(img_gray, "bounding_box")
    writeImage(original, "box_mainImg")

def opening(binary):
    erode = erosion(binary)
    dilate = dilation(erode)
    
    opened_Img = np.copy(dilate)
    return opened_Img

def dilation(img):
    
    dilated_img = np.copy(img)
    
    for i in range(1, img.shape[0]-2):
        for j in range(1, img.shape[1]-2):
            if img[i,j] == 255:
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

def closing(binary):

    dilate = dilation(binary)
    erode = erosion(dilate)
    
    closed_Img = np.copy(erode)
    
    return closed_Img

def open_close(img):
    opened = opening(img)
    closed = closing(opened)
    
    return closed

    
def pointDetection(img):
    
    original = np.copy(img)
    pointDetected = np.zeros(original.shape)
    
#    mask = np.array([[-1, -1, -1, -1, 2], [-1, -1, -1, 2, -1], [-1, -1, 12, -1, -1], [-1, 2, -1, -1, -1], [2, -1, -1, -1, -1]])
    mask = createKernel(5)
    
    thresh = 2300
    
    x = math.floor(mask.shape[0]/2)
    y = math.floor(mask.shape[1]/2)
    
    for i in range ( x, original.shape[0] - x):
        for j in range ( y, original.shape[1] - y):
            newSum = 0
            for u in range ( mask.shape[0] ):
                for v in range ( mask.shape[1] ): 
                   newSum = newSum + mask[u, v] * img[i+u-x, j+v-y]    
            
            pointDetected[i, j] = newSum
            
    for i in range ( x, pointDetected.shape[0] - x ):
        for j in range ( y, pointDetected.shape[1] - y ):
            if abs(pointDetected[i,j]) < thresh:
                pointDetected[i,j] = int(0)
            else:
                pointDetected[i,j] = int(255)
                
    
    onlyPoint = dilation (pointDetected)
    print("\nCoordinates of the point = ",np.argwhere(pointDetected == np.max(pointDetected)))
    
    return onlyPoint, thresh

img_gray = cv2.imread("original_imgs/segment.jpg")

def task2():
    
    global t
    
    image = cv2.imread("original_imgs/point.jpg", 0)
    
    withPoint, thresh = pointDetection(image)
    writeImage(withPoint, "point_detected")
    print("\nPoint successfully found in the image.Threshold = ", thresh)
    print("\ndetected_point.jpg written to disk. ")
    

########################### PART 2 #################################################################################
    
    segment = cv2.imread("original_imgs/segment.jpg", 0)
    
    h_thresh, t = heuristicThreshold(segment)
    writeImage(h_thresh, "bones_segmented")
    print ("\nBones successfully segmented. Threshold = ", t)
    print ("\nbones_segmented.jpg written to disk. ")
    
    global box
    box = h_thresh
    
    bounding_Box (box)
    
    print("Time taken:", time.time()-t," seconds")


if __name__=='__main__':
    try:
        t = time.time()
        task2()
    except:
        pass