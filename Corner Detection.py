# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 20:10:29 2019

@author: shrey
"""

import cv2
import numpy as np
import sys
from scipy import signal


def main():
   
   #HARRIS CORNER DETECTOR
   
   #LOAD IMAGE
   input_img = cv2.imread(r'C:\Users\shrey\OneDrive\Documents\CV HW\HW 2\hotel.seq0.png', cv2.IMREAD_GRAYSCALE)
   
   #OBTAIN X AND Y GRADIENTS
   Ix = cv2.Sobel(input_img, cv2.CV_64F, 1, 0, ksize=3)
   Iy = cv2.Sobel(input_img, cv2.CV_64F, 0, 1, ksize=3)
   Ixx = np.multiply(Ix, Ix)
   Ixy = np.multiply(Ix, Iy)
   Iyy = np.multiply(Iy, Iy)
   
   #GAUSSIAN FILTER INDIVIDUAL IMAGES
   Sxx = cv2.GaussianBlur(Ixx,(5,5),3)
   Sxy = cv2.GaussianBlur(Ixy,(5,5),3)
   Syy = cv2.GaussianBlur(Iyy,(5,5),3)
   #M = np.array ([(Sxx,Sxy),(Sxy,Syy)])
   
   #DEFINE CORNER STRENGTH FUNCTION
   det = (Sxx * Syy) - (Sxy**2)
   trace = Sxx + Syy
   r = det - 0.05*(trace**2)
   
   cnt = 0
   #RESULT IS DIALATED TO OPTIMIZE CORNERS
   dst = cv2.dilate(r,None)
   
   ## NON MAXIMAL SUPPRESSION WITH 5X5 WINDOW
   
   rows = dst.shape[0]
   cols = dst.shape[1]
   for i in range(3,rows-3,5):
       for j in range(3,cols-3,5):
           #print (dst[i,j])
           # getNeighbours
           maxval = -1*sys.maxsize
           for x in range (i-2,i+3):
               for y in range(j-2,j+3):
                   if dst[x,y] > maxval:
                       maxval = dst[x,y]
           for x in range (i-2,i+3):
               for y in range(j-2,j+3):
                   if dst[x,y] < maxval:
                       dst[x,y] = 0
           
           

           
   # SET THRESHOLD
   out_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
   out_img[dst>0.01*dst.max()]=[0,255,0]
   
   #DISPLAY IMAGE
   cv2.imshow('KEYPOINTS',out_img)
   cv2.imwrite('KEYPOINTS.jpeg',out_img)
   cv2.waitKey(0)
   
   cv2.destroyAllWindows()
   
       

if __name__== "__main__":
  main()