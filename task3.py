# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 22:13:25 2018

''''''''''' CVIP PROJECT 3 - TASK 3 '''''''''''''''''''''''''''''
@author: ANIRUDDHA SINHA
UBIT : asinha6
Person #: 50289428

"""

import cv2
import numpy as np
import math
import time

houghRGB = []
img_diagonal_length = 0


'function to write the image to disk'
def writeImage(img, imageName):
    cv2.imwrite("results/" + imageName + ".jpg", img)


def toBinary(img, thresh):
    retBinary, binary_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return binary_img 

def edgeDetection(original, thresh):
    
    img = np.copy(original)
    h, w = img.shape
    
    img = cv2.GaussianBlur(img,(5,5),0)
    
    sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    
    x , y = sobel_x.shape
    x0 = math.floor (x/2)
    y0 = math.floor (y/2)
    
    edge_img = np.zeros(img.shape)    

    for i in range ( x0 , h-x0 ):
        for j in range( y0 , w-y0):
            temp_x = 0
            temp_y = 0
            for u in range (x):
                for v in range (y): 
                    temp_x += sobel_x [ x-u-x0, y-v-y0 ] * img [ i+u-x0 , j+v-y0 ]    
                    temp_y += sobel_y [ x-u-x0, y-v-y0 ] * img [ i+u-x0 , j+v-y0 ]
                    
            edge_img [ i, j ] = math.sqrt( temp_x**2 + temp_y**2)

    return toBinary(edge_img, thresh)      

def groupPoints_byAngle (coordinates, linesAngles):
    
    group1 = []
    group2 = []
    
    for point in coordinates:
        if point[1] in linesAngles[0]:
            group1.append(point)
        elif point[1] in linesAngles[1]:
            group2.append(point)
     
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    norm_Factor = np.array([100, 45])

    new_list1 = []
    temp_index = 1
    
    counter = 0
    for point in group1:
        counter+=1
        d = int (abs(point[0]/norm_Factor[0]))
        
        if temp_index != d:
            temp_index = d
            temp_list = []
            temp_list.append((point[0], point[1]))
            new_list1.append(temp_list)
        
        else:
            temp_list = new_list1[d]
            temp_list.append((point[0], point[1]))
            new_list1[d] = temp_list
        
    new_list1 = np.array(new_list1)
    
    new_list2 = {}
    
    for point in group2:
        d = math.floor (point[0]/norm_Factor[1])
        
        if new_list2.get(d) is None:
#            temp_index = d
            temp_list = []
            temp_list.append((point[0], point[1]))
            new_list2[d] = temp_list
        
        else:
            temp_list = new_list2.get(d)
            temp_list.append((point[0], point[1]))
            new_list2[d] = temp_list
    
    new_list2 = new_list2.values()
    
    verticalLines_Points = []
    slantLines_Points = []
    
    for item in new_list1:
        temp_point1 = item [ math.ceil ((len(item)-1)/2) ]
        verticalLines_Points.append(temp_point1)
        
    for item in new_list2:
        temp_point2 = item [ math.ceil ((len(item)-1)/2) ]
        slantLines_Points.append(temp_point2)
    
    verticalLines_Points = np.array (verticalLines_Points)
    slantLines_Points = np.array (slantLines_Points)
    
    return verticalLines_Points, slantLines_Points

def drawLines (points, original_colour, typeLine, C):
    
    img = np.copy(original_colour)
    
    c = 0

    for item in points:

        rho, theta = item  
        
        c += 1
        
        if typeLine == "red_line":
            if c == 2:
                rho += 3
                theta = -2   
            elif c == 3 or c==4:
                rho += 2
                theta = -2.5
            elif c == 5 or c == 6:
                rho -= 1.5
                theta = -2.5
        
        theta = np.radians (theta)
        
        a = np.cos(theta) 
#        print ("a = ", a)
        b = np.sin(theta)
#        print ("b = ", b)
     
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
#        print ("x1 = ", x1)
        y1 = int(y0 + 1000*(a))
#        print ("y1 = ", y1)
        x2 = int(x0 - 1000*(-b))
#        print ("x2 = ", x2)
        y2 = int(y0 - 1000*(a))
#        print ("y2 = ", y2)
    
        cv2.line(img,(x1,y1),(x2,y2),C,2)
    
    writeImage (img, typeLine)
    print("\n" + typeLine + ".jpg written to disk.")

def HoughLines (accumulator):
    
    global houghRGB
    
    lines = np.argwhere((accumulator>110) & (accumulator<180))

    coordinates = np.zeros(lines.shape)
    coordinates[:,0] = lines[:,0] - img_diagonal_length
    coordinates[:,1] = lines[:,1] - 90
    
    linesAngles = np.array([[-2.0, -3.0], [-36.0]])

    points_vertical, points_slanted = groupPoints_byAngle (coordinates, linesAngles)
    drawLines (points_vertical, houghRGB, "red_line", (0,0,255))
    drawLines (points_slanted, houghRGB, "blue_lines", (255, 0, 0))
    
    return houghRGB

def lines_votingHough (original):
    
    img = np.copy (original)
    print (img.shape)
    
    global img_diagonal_length
    img_diagonal_length = int( np.sqrt( img.shape[0]**2 + img.shape[1]**2 ) )
    print ("\nImage diagonal's length = ", img_diagonal_length)
    
    theta = np.linspace (-90, 90, 181)
    
    accumulator = np.zeros(( 2*img_diagonal_length , theta.shape[0]), dtype = np.uint64 )
    print (accumulator.shape)

    counter = 1
    for y in range (img.shape[0]):
        for x in range(img.shape[1]):
            
            if img[y, x] == 255:
                for angle in range (accumulator.shape[1]):
                    angle_rad = np.radians (theta[angle])
                   
                    p = int(x*np.cos(angle_rad) + y*np.sin(angle_rad)) + img_diagonal_length
                    accumulator [ p, angle ] += 1
                    counter += 1

    print ("\nMax value in lines accumulator = ", np.max(accumulator))

    writeImage (accumulator, "line_accumulator")
    print ("\nLines Accumulator image written to disk.")
    
    return accumulator

 
def detectHoughLines (original):
    
    img = np.copy (original)
    
    accumulator = lines_votingHough (img)
    
    HoughLines (accumulator)

def circle_HoughVoting (edge_img):
    
    global img_diagonal_length
    angle_Range = np.arange(0,361)
    row, cols = edge_img.shape
    
    ab_Space = np.zeros((row, cols, img_diagonal_length))
    print (ab_Space.shape)
    
    radii = np.arange(22,26)
    
    for y in range(ab_Space.shape[0]-2):
        for x in range(ab_Space.shape[1]-2):
            
            if edge_img[y,x] == 255:
                
                for R in radii:
                    for angle in angle_Range:
                        
                        a = int (x - R*np.cos(np.radians(angle)))
                        b = int (y - R*np.sin(np.radians(angle)))
                        
                        if (b >= ab_Space.shape[0]) or (a >= ab_Space.shape[1]) :
                            continue
                        else:
                            ab_Space [b,a, R] += 1
                     
#    print ("max =", np.max(ab_Space))                    
    
    return ab_Space

def draw_Circles (circle_points):
    global houghRGB
    
    for item in circle_points:
        i = item[0]
        j = item[1]
        center = (j,i)
        
        R = item[2]
        cv2.circle(houghRGB, center, R, (0,255,255))

    writeImage(houghRGB, "circle")
    
def bonus_detectCircles (original):

    img = np.copy(original)
    
    ab_Space = circle_HoughVoting (img)
    circle_points = np.argwhere (ab_Space>280)
    
    draw_Circles (circle_points)
    print("\nCircles successfully drawn. ")
    print("\ncircles.jpg written to disk. ")
               
def task3():
    
    global t
    
    print("\n=========== STARTING EXECUTION =====================")
    
    global houghRGB
    houghRGB = cv2.imread("original_imgs/hough.jpg")
    hough = cv2.imread("original_imgs/hough.jpg", 0)
    
    sobel_Edge_lines = edgeDetection(hough, 100)
#    writeImage(sobel_Edge_lines, "SobelEdge_Hough")
#    print ("\nSobel Edge image written to disk.")
    
    detectHoughLines (sobel_Edge_lines)
    
    sobel_Edge_circles = edgeDetection(hough, 60)
    bonus_detectCircles (sobel_Edge_circles)
    
    print("\nTime taken:", time.time()-t," seconds")

if __name__=='__main__':
    try:
        t = time.time()
        task3()
    except:
        pass