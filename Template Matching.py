# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 01:06:01 2019

@author: shrey
"""

import cv2
import numpy as np
from skimage import img_as_ubyte
import xlsxwriter


################################
#noisy - modified from Shubham Pachori on stackoverflow
def noisy(image, noise_type, sigma):
    if noise_type == "gauss":
        row,col = image.shape
        mean = 0
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_type == "s&p":
        row,col = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
        for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
        for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type =="speckle":
        row,col = image.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)
        noisy = image + image * gauss
        return noisy
    
def match(input_img,temp):
   
    # ADD DESIRED NOISE LEVELS AND GAUSSIAN FILTER IN LOOP
   l = ["gauss"]   
   coff_list = []
   for i in range(0,11):
       
       for j in range(0,6):  
           n= 6*j +1
           noisy_img = np.uint8(noisy(input_img, l[0], i))
           smoothed_img = cv2.GaussianBlur(noisy_img,(n,n),j)
           #smoothed_temp = cv2.GaussianBlur(temp,(n,n),j)
           output_img = cv2.matchTemplate(smoothed_img,temp,cv2.TM_CCOEFF_NORMED)
           out= img_as_ubyte(output_img)
           coff=np.amax(output_img)       
           coff_list.append(coff)
          
           smoothed_img1 = img_as_ubyte(smoothed_img)
           noisy_img1 = img_as_ubyte (noisy_img)
           j = (j+1) % len(l)
   return (out, noisy_img1, smoothed_img1, coff_list)



def main():
   
   # LOAD IMAGE AND TEMPLATE
   input_img = cv2.imread(r'C:\Users\shrey\OneDrive\Documents\CV HW\HW 2\motherboard-gray.png', cv2.IMREAD_GRAYSCALE)
   temp = cv2.imread(r'C:\Users\shrey\OneDrive\Documents\CV HW\HW 2\template.png', cv2.IMREAD_GRAYSCALE)
   
   #OBTAIN DESIRED MATCHING
   l1 = match(input_img, temp)
   out1 = l1[0]
   input_img1 = input_img
   noisy_img1 = l1[1]
   smoothed_img1 = l1[2]
   
   # DISPLAY CORRELATION-COEFFICIENTS AS EXCEL ARRAY
   coff_l = l1[3]
   c=np.reshape(coff_l, (11, 6))
   print (c)
   workbook = xlsxwriter.Workbook('Correlations.xlsx')
   worksheet = workbook.add_worksheet()
   col = 0
   for row, data in enumerate(c):
        worksheet.write_row(row, col, data)
   workbook.close()
   
   #dISPLAY AND SAVE LAST OUTPUTS (NOISE 10 SIGMA 5)
   cv2.imshow('original',input_img1)
   cv2.imshow('output',out1)
   cv2.imshow('Noisy',noisy_img1)
   cv2.imshow('Smoothed',smoothed_img1)
   cv2.imwrite('original.jpeg',input_img1)
   cv2.imwrite('output.jpeg',out1)
   cv2.imwrite('Noisy.jpeg',noisy_img1)
   cv2.imwrite('Smoothed.jpeg',smoothed_img1)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
                
         
if __name__== "__main__":
  main()