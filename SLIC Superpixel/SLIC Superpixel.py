# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:01:54 2019

@author: shrey
"""
import math
from skimage import color
import numpy as np
import cv2


def CreateSuperPixel(h, w,img):
    return SuperPixels(h, w,img[h,w][0],img[h,w][1],img[h,w][2])

# INITIALIZE CLUSTERS CENTERS
def InitialClusterCenter(S,img,img_h,img_w,clusters):
    h = (S // 2)
    w = (S // 2)
    while h < img_h:
        while w < img_w:
            clusters.append(CreateSuperPixel(h, w,img))
            w += S
        w = S // 2
        h += S
    return clusters

def FindGradient(h, w,img,img_w,img_h):
    
    if w + 1 >= img_w:
        w = img_w - 2
    if h + 1 >= img_h:
        h = img_h - 2
           
    grad = img[w + 1, h + 1][0] - img[w, h][0] + img[((w + 1) % img_w), ((h + 1) % img_h)][1] - img[w, h][1] + img[w + 1, h + 1][2] - img[w, h][2]

    return grad

# MOVE CLUSTER CENTERS
def OptimizeClusterCenter(clusters,img):
    for c in clusters:
        cluster_gradient = FindGradient(c.h, c.w,img,img_w,img_h)
        
        for dh in range(-1, 2):
            for dw in range(-1, 2):
                H = c.h + dh
                W = c.w + dw
                new_gradient = FindGradient(H,W, img,img_w,img_h)
                if new_gradient < cluster_gradient:
                    c.update(H, W,img[H,W][0], img[H,W][1],img[H,W][2])
                    
    

def ClusterPixels(clusters,S,img,img_h,img_w,tag,dis):
    for c in clusters:
        for h in range(c.h - 2 * S, c.h + 2 * S):
            if h < 0 or h >= img_h: continue
            for w in range(c.w - 2 * S, c.w + 2 * S):
                if w < 0 or w >= img_w: continue
                l, a, b = img[h,w]
                Dc = math.sqrt(math.pow(l - c.l, 2) + math.pow(a - c.a, 2) + math.pow(b - c.b, 2))
                Ds = math.sqrt(math.pow(h - c.h, 2) + math.pow(w - c.w, 2))
                D = math.sqrt(math.pow(Dc / m, 2) + math.pow(Ds /S, 2))
                if D < dis[h,w]:
                    if (h, w) not in tag:
                        tag[(h, w)] = c
                        c.pixels.append((h, w))
                    else:
                        tag[(h, w)].pixels.remove((h, w))
                        tag[(h, w)] = c
                        c.pixels.append((h, w))
                    dis[h, w] = D
                    

def UpdateMean(clusters):
    for c in clusters:
        sum_h = sum_w = number = 0
       
        for p in c.pixels:
            sum_h += p[0]
            sum_w += p[1]
            number += 1
            H = sum_h // number
            W = sum_w // number
            c.update(H, W,img[H, W][0], img[H, W][1], img[H, W][2])
       
   

# DRAW CONTOURS AROUND SUPERPIXELS
def DrawContours(img,name,clusters):
    
    mask = np.zeros(img.shape[:2], dtype = "uint8")
    for c in clusters:
        for p in c.pixels:
            mask[p[0],p[1]]= 255
        image1, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        image = cv2.drawContours(imgR, contours, -1, (0,0,0), 1)
        mask = np.zeros(img.shape[:2], dtype = "uint8")
    res = image
    cv2.imwrite(name, res)

            
# SLIC
def Slic(S,img,img_h,img_w,clusters,tag,dis):
    clusters = InitialClusterCenter(S,img,img_h,img_w,clusters)
#    OptimizeClusterCenter(clusters,img)
#    
#    for i in range(10): # usually the algortihm converges within 10 iterations
#        ClusterPixels(clusters,S,img,img_h,img_w,tag,dis)
#        UpdateMean(clusters)
#        if i == 9 : # to print the output after 10 iterations
#            name = 'out_m{m}_k{k}.png'.format(loop=i, m=m, k=k)
#            DrawContours(img,name, clusters)
#            print('Saved Superpixel Image')
    return clusters



class SuperPixels(object):

    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)
        self.pixels = []
        
    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b
       
        
# read the input RGB image
rgb = cv2.imread("C:\\Users\\shrey\\OneDrive\\Documents\\CV HW\\HW EC1\\spock.png",cv2.IMREAD_UNCHANGED)
print('Reading Image...')

# input images are resized for processing

#imgR = cv2.resize(rgb,(650,650))
imgR=rgb
# convert RGB to LAB
img = color.rgb2lab(imgR)
print('Coverting to lab color space...')

k = 256   # Number of Super pixels
m = 20    # Constant for normalizing the color proximity, range of m = [1,40]

img_h = img.shape[0] # Image Height
img_w = img.shape[1] # Image Width

N = img_h * img_w  # Total number of pixels in the image
S = int(math.sqrt(N /k)) # average size of each superpixel

clusters = []
tag = {}
# initialize the distance between pixels and cluster center as infinity
dis = np.full((img_h, img_w), np.inf)

print('Making superpixels....')
cluster = Slic(S,img,img_h,img_w,clusters,tag,dis)
