# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:27:38 2019

@author: shrey
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:28:58 2019

@author: shrey
"""
import numpy as np
import cv2
import time
import scipy.ndimage as ndimg
from sklearn.cluster import KMeans
import argparse

NROT = 6
NPER = 8
NFILT = NROT*NPER
FILTSIZE = 49
NCLUSTERS = 8
TEXELSIZE = 4
pathName = "C:\\Users\\shrey\\OneDrive\\Documents\\CV HW\\HW 4\\IMAGES\\"
#fileName = "aerial-houses"
#fileName = "texture"
fileName = "selfie"
def saveImage(img, name):
    cv2.imwrite(pathName + name + ".png", img)
    return

# MAKE FILTERS
    
def gaussian1d(sigma, mean, x, ord):
    x = np.array(x)
    x_ = x - mean
    var = sigma**2

    # Gaussian Function
    g1 = (1/np.sqrt(2*np.pi*var))*(np.exp((-1*x_*x_)/(2*var)))

    if ord == 0:
        g = g1
        return g
    elif ord == 1:
        g = -g1*((x_)/(var))
        return g
    else:
        g = g1*(((x_*x_) - var)/(var**2))
        return g

def gaussian2d(sup, scales):
    var = scales * scales
    shape = (sup,sup)
    n,m = [(i - 1)/2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    return g

def log2d(sup, scales):
    var = scales * scales
    shape = (sup,sup)
    n,m = [(i - 1)/2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    h = g*((x*x + y*y) - var)/(var**2)
    return h

def makefilter(scale, phasex, phasey, pts, sup):

    gx = gaussian1d(3*scale, 0, pts[0,...], phasex)
    gy = gaussian1d(scale,   0, pts[1,...], phasey)

    image = gx*gy

    image = np.reshape(image,(sup,sup))
    return image

def makeLMfilters():
    sup     = 49
    scalex  = np.sqrt(2) * np.array([1,2,3])
    norient = 6
    nrotinv = 12

    nbar  = len(scalex)*norient
    nedge = len(scalex)*norient
    nf    = nbar+nedge+nrotinv
    resF     = np.zeros([sup,sup,nf])
    hsup  = (sup - 1)/2

    x = [np.arange(-hsup,hsup+1)]
    y = [np.arange(-hsup,hsup+1)]

    [x,y] = np.meshgrid(x,y)

    orgpts = [x.flatten(), y.flatten()]
    orgpts = np.array(orgpts)

    count = 0
    for scale in range(len(scalex)):
        for orient in range(norient):
            angle = (np.pi * orient)/norient
            c = np.cos(angle)
            s = np.sin(angle)
            rotpts = [[c+0,-s+0],[s+0,c+0]]
            rotpts = np.array(rotpts)
            rotpts = np.dot(rotpts,orgpts)
            resF[:,:,count] = makefilter(scalex[scale], 0, 1, rotpts, sup)
            resF[:,:,count+nedge] = makefilter(scalex[scale], 0, 2, rotpts, sup)
            count = count + 1

    count = nbar+nedge
    scales = np.sqrt(2) * np.array([1,2,3,4])

    for i in range(len(scales)):
        resF[:,:,count]   = gaussian2d(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        resF[:,:,count] = log2d(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        resF[:,:,count] = log2d(sup, 3*scales[i])
        count = count + 1

    return resF

#SAVE FILTERS
    
def saveFilters(img):
    (height, width, depth) = img.shape
    count = 0
    for row in range(NPER):
        for col in range(NROT):
            tempImg = img[:, :, count]
            filename = "Filters\\LM_" + str(row) + "_" + str(col)
            normedFilter = normImg(tempImg)
            saveImage(normedFilter, filename)
            count = count + 1
    return

#NORMALIZE FUNCTION

def normImg(img):
    tempImg = np.zeros_like(img)
    tempImg = (cv2.normalize(img, tempImg, 0.0, 127.0, cv2.NORM_MINMAX))
    res = (tempImg+128.0).astype(np.uint8)
    return res

#MAKE MOSAIC OF FILTERS

def makeMosaic(img):
    (height, width, depth) = img.shape
    res = np.zeros((height*8, width*6), np.float64)
    count = 0
    for row in range(8):
        for col in range(6):
            res[row*height:(row+1)*height, col*width:(col+1)*width] = \
            normImg(img[:, :, count])
            count = count + 1
    return res

# APPLY LM FILTERS TO INPUT IMAGE 
    
def applyLMfilters(inputImg, filt):
    
    reslist = []
    mean , sd = inputImg.mean() , inputImg.std()
    img = (inputImg-mean)/sd
    img1= img.astype(np.float32)
    for i in range (NFILT):
        filti = filt[:,:,i]
        res= cv2.filter2D(img1,-1,filti)
        reslist.append(res)
    resA = np.stack(reslist,axis=2)
    return resA

# FORM TEXELS
    
def SliceTexel(in_arr, sz):
    
    out_arr = []
    rows = in_arr.shape[0]
    cols = in_arr.shape[1]
    
    for i in range(0,rows,sz):
        l1 = []
        for j in range(0,cols,sz):
            c=in_arr[i:i+sz,j:j+sz]
            l1.append(np.full(c.shape,np.mean(in_arr[i:i+sz,j:j+sz])))
        out_arr.append(l1)
        
    #PUT THE ARRAY BACK TOGTHER
    out_arr1 = []
    for i in out_arr:
        m = i[0]
        for j in range(len(i)-1):
            m = np.concatenate((m,i[j+1]),1)
        out_arr1.append(m)
    arr_0 = out_arr1[0]
    for i in range(len(out_arr1)-1):        
        arr_0 = np.append(arr_0, out_arr1[i+1],axis=0)
    return arr_0



def formTexels(R,sz):

    Texlist =[]
    for k in range (NFILT):
        RS= R[:,:,k]
        EachResponse = SliceTexel(RS,sz)
        Texlist.append(EachResponse) 
                
    Texel = np.stack(Texlist,axis=2)
#    print(Texel[0:20,0:20,1])
    return Texel
   
    
    
# CLUSTER TEXELS USING KMEANS

def segmentKMeans(R, nclus):

    height,width,depth = R.shape
    RClus = np.reshape(R,(height*width,depth))
    Kmeans = KMeans(n_clusters=nclus,random_state=0).fit(RClus)
    label = Kmeans.labels_
    print(np.unique(label))
    res=label.reshape(height,width)
    res=(res/3)*255
    res2D=res.astype(np.uint8)  
    PsuedoCol=cv2.applyColorMap(res2D, cv2.COLORMAP_JET)
    
    return PsuedoCol

# MAIN
    
parser = argparse.ArgumentParser();
parser.add_argument('--image_path', required=True, help='Absolute path of the image to be used.');
if __name__ == '__main__':
    args = parser.parse_args();
    pathName = args.image_path;
    print('IMAGE PATH: ', pathName);

currTime = time.time()

# CALL MAKE FILTER
F = makeLMfilters()
saveFilters(F)
saveImage(makeMosaic(F), "allFilters")

# LOAD IMAGE
inputImage = cv2.cvtColor(cv2.imread(pathName + fileName + ".png"), cv2.COLOR_BGR2GRAY)

# FIND FILTER RESPONSE

rawR = applyLMfilters(inputImage, F)
if (True):
    R = formTexels(rawR, TEXELSIZE)
else:
    R = rawR

# SEGMENTING
    
pcolor = segmentKMeans(R, NCLUSTERS)
saveImage(pcolor, fileName+"_Seg_"+str(NCLUSTERS))
elapsedTime = time.time() - currTime
print("Completed; elapsed time = ", elapsedTime)
    

