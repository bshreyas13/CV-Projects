# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:23:19 2019

@author: shrey
"""

import numpy as np
import cv2
import math

pathName = 'C:\\Users\\shrey\\OneDrive\\Documents\\CV HW\\HW3\\ECE5554 FA19 HW3 images\\'
inputImage = cv2.imread(pathName + 'VAoutline.png', cv2.IMREAD_GRAYSCALE)
name= 'Contour{!s}'
thresh = 70;
binary = cv2.threshold(inputImage, thresh, 255, cv2.THRESH_BINARY)[1]
(height, width) = binary.shape

def ShowImage(image,name):
    cv2.imshow(name, image)
    return
def SaveImage(img, name):
    cv2.imwrite(pathName + name + ".png", img)
    return


#FIND START POINT
ystt = np.uint8(height/2) # look midway up the image
for xstt in range(width): # from the left
 if (binary[ystt, xstt] > 0):
     break
startPoint= [ystt, xstt]
#print (type(startPoint))
CtrPoints=[startPoint]

##PAVLIDIS START

# CHECK PIXELS FOR FOREGROUND
def LookAround(Point,O):
    
    [y,x]= Point
    orientation ={1:([y-1,x-1]),
             2:([y-1,x+1]),
             3:([y+1,x+1]),
             4:([y+1,x-1])}
    if O == 1:
        [i,j]=orientation[O] 
        if binary[i,j]>0:
            if O==1:
                newO=4
            else:
                newO=O-1
            newPoint=[i,j]
            newPoint=[i,j] 
            return(newO,newPoint)
        elif binary[i,j+1]>0:
            newO=O
            newPoint=[i,j+1] 
            return(newO,newPoint)
        elif binary[i,j+2]>0:
            newO=O
            newPoint=[i,j+2] 
            return(newO,newPoint)
        else:
            newO=(O % 4)+1
            newPoint=Point 
            return(newO,newPoint)
            
    elif O==2:
        [i,j]=orientation[O] 
        if binary[i,j]>0:
            if O==1:
                newO=4
            else:
                newO=O-1
            newPoint=[i,j] 
            return(newO,newPoint)
        elif binary[i+1,j]>0:
            newO=O
            newPoint=[i+1,j] 
            return(newO,newPoint)
        elif binary[i+2,j]>0:
            newO=O
            newPoint=[i+2,j] 
            return(newO,newPoint)
        else:
            newO=(O % 4)+1
            newPoint=Point 
            return(newO,newPoint)
            
    elif O==3:
        [i,j]=orientation[O] 
        if binary[i,j]>0:
            if O==1:
                newO=4
            else:
                newO=O-1
            newPoint=[i,j] 
            return(newO,newPoint)
        
        elif binary[i-1,j]>0:
            newO=O
            newPoint=[i-1,j] 
            return(newO,newPoint)
        elif binary[i-1,j]>0:
            newO=O
            newPoint=[i-1,j] 
            return(newO,newPoint)
        else:
            newO=(O % 4)+1
            newPoint=Point 
            return(newO,newPoint)
    elif O==4:
        [i,j]=orientation[O] 
        if binary[i,j]>0:
            if O==1:
                newO=4
            else:
                newO=O-1
            newPoint=[i,j] 
            return(newO,newPoint)
        elif binary[i-1,j]>0:
            newO=O
            newPoint=[i-1,j] 
            return(newO,newPoint)
        elif binary[i-2,j]>0:
            newO=O
            newPoint=[i-2,j] 
            return(newO,newPoint)
        else:
            newO=(O % 4)+1
            newPoint=Point 
            return(newO,newPoint)

#FIND CONTOUR POINTS AND STOP AT STARTING POINT 
def FindContour(nextPoint,nextO):
    
    [y1,x1]=nextPoint
    while [y1,x1] != startPoint :
            (nextO,nextPoint)=LookAround(nextPoint,nextO)
            [y1,x1]=nextPoint
            if nextPoint not in (CtrPoints):
                CtrPoints.append(nextPoint)
            FindContour(nextPoint,nextO)
            return (CtrPoints)



def Pavlidis(Point):
    # INITIALIZE AND SHOW TRACED CONTOUR 
    [a,b]=LookAround(Point,1)
    C2=FindContour(b,a)
    CtrPoints1 = np.asarray(C2)
    showContour(CtrPoints1,np.zeros_like(inputImage),'Contour')
    return CtrPoints1

# DRAW LINES BETWEEN CONTOUR POINTS AND DISPLAY 
def showContour(ctr,img, name):
    
 contourImage = img

 length = ctr.shape[0]


 for count in range(length):
     contourImage[ctr[count, 0], ctr[count, 1]] = 255
     
     Cnt=(count+1)%length
     cv2.line(contourImage,(ctr[count, 1], ctr[count, 0]),\
              (ctr[Cnt, 1], ctr[Cnt, 0]),
              (128,128,128),1)
 ShowImage(contourImage, name)
 SaveImage(contourImage, name)

    
#ONE PASS DCE

def onePassDCE(ContourPoints):
    
    x_list = []
    y_list = []
    k_list = []
    
    length = len(ContourPoints)
    #print(length)
    UpdateCtr = ContourPoints.tolist()
    
    
    # UNPACK X AND Y COORDINATES
    for l in range (length):
        [y2,x2]= UpdateCtr[l]
        #print([y2,x2])
        x_list.append(x2)
        y_list.append(y2)
        
    
    y=y_list
    x=x_list
    
    #FIND K FOR EACH CONTOUR POINT
    for i in range(length):
        
        
        I= (i+1)%length
        y4= y[I] - y[i]
        x4= x[I] - x[i]
        
        y3= y[i] - y[i-1]
        x3= x[i] - x[i-1]
        
        length3 = math.sqrt(math.pow((x[i-1]-x[i]),2) + math.pow((y[i-1]-y[i]),2))
        length4 = math.sqrt(math.pow((x[I]-x[i]),2) + math.pow((y[I]-y[i]),2))
    
        k1= np.arctan2(y3,x3)
        k2= np.arctan2(y4,x4)
        k = abs(k1 - k2)*length3*length4/(length3+length4)
        
        k_list.append(k)
        
    #FIND LOWEST K
    kminOnePass = min(k_list)
    #print('k',kminOnePass)
    
    #REMOVE POINTS CORRESPONDING TO LOWEST K
    
    checkLength = len(ContourPoints)
    
    for kidx in range(checkLength):
        if kminOnePass == k_list[kidx]:     
            remK= kidx
            UpdateCtr.remove(UpdateCtr[remK])
            UpdateCtr1= np.asarray(UpdateCtr)
            k_list.remove(k_list[kidx]) 
            checkLength=len(UpdateCtr)
            break
            #print('ctr',checkLength)
            
    return(UpdateCtr1)
    
def GaussArea(ContourPoints):
    a=0
    check = (ContourPoints.shape[0])
    for ind in range(check):
        [y5,x5] = ContourPoints[ind]
        [y6,x6] = ContourPoints[(ind+1)%check]
        a += (x5*y6) - (y5*x6) 
    area = abs(a/2)
    return area
        
    


Ctr=Pavlidis(startPoint)
print(Ctr.shape, GaussArea(Ctr))
UpdateCtr = Ctr
levels = 1

for step in range(6):
     numLoops = math.floor(UpdateCtr.shape[0]/2)
#     print(numLoops)
#     print(len(UpdateCtr))
     for idx in range(numLoops):
         UpdateCtr=onePassDCE(UpdateCtr)
     showContour(UpdateCtr,np.zeros_like(inputImage), name.format(levels))
     print(numLoops, UpdateCtr.shape, GaussArea(UpdateCtr))
     levels += 1     
cv2.waitKey(0)
cv2.destroyAllWindows()
           
    
    