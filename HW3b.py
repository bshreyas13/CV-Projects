# -*- coding: utf-8 -*-
"""
Created on Tue Nov 07 11:31:57 2019

@author: shrey
"""

import cv2
import numpy as np

img_ = cv2.imread('C:\\Users\\shrey\\OneDrive\\Documents\\CV HW\\HW3\\panorama-stitching\\images\\Goodwin2.png')
img = cv2.imread('C:\\Users\\shrey\\OneDrive\\Documents\\CV HW\\HW3\\panorama-stitching\\images\\Goodwin1.png')
img3 = cv2.imread('C:\\Users\\shrey\\OneDrive\\Documents\\CV HW\\HW3\\panorama-stitching\\images\\Goodwin0.png')
#img_ = cv2.resize(img_, (0,0), fx=0.5, fy=0.5)
#img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
#img3 = cv2.resize(img3, (0,0), fx=0.5, fy=0.5)
#TRIM EXTRA BLACK


def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:]) 
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])    
    return frame

def stitch(img_, img):
    
    img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)

    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()

    # find key points
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1,des2,None)

    matches.sort(key=lambda x: x.distance, reverse=False)

    numGoodMatches = int(len(matches) * 0.03)
    good = matches[:numGoodMatches]
 
    draw_params = dict(matchColor=(0,255,0),
                       singlePointColor=None,
                       flags=2)
    img3 = cv2.drawMatches(img_,kp1,img,kp2,good,None,**draw_params)
    cv2.imshow("original_image_drawMatches.jpg", img3)

    #STITCH
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        h,w = img1.shape
        print (M)
        
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        cv2.imshow("original_image_overlapping.jpg", img2)
    else:
        print("Not enought matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))

    #WARP IMAGES AND DISPLAY THEM AFTER STITCHING
    dst = cv2.warpPerspective(img_,M,(img.shape[1] + img_.shape[1], img.shape[0]))
    dst[0:img.shape[0],0:img.shape[1]] = img
    cv2.imshow("original_image_stitched.jpg", dst)
    
    cv2.imshow("original_image_stitched_crop", trim(dst))
    cv2.imwrite("Panostiched Goodwin .png", trim(dst))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return(dst)

imgInt = stitch (img_,img)

imgFinal = stitch (imgInt,img3)


