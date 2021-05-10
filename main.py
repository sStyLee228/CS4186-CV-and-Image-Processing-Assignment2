import math
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

image_folder_path = 'C:\\Users\\Aslan\\Desktop\\CS4186 assignment 2\\StereoMatchingTestings'
images_paths = [os.path.join(root, f) for root, dir, files in os.walk(image_folder_path) for f in files if 'disp' not in f]
images_paths = [images_paths[i:i+2] for i in range(0, 5, 2)]
images = [[cv.imread(sublist[0], cv.IMREAD_GRAYSCALE), cv.imread(sublist[1], cv.IMREAD_GRAYSCALE)] for sublist in images_paths]

im1, im2 = images[0][0], images[0][1]

print(im1.shape, im2.shape)

h, w = im1.shape

# win_size = 5
# min_disp = -1
# max_disp = 63 #min_disp * 9
# num_disp = max_disp - min_disp # Needs to be divisible by 16
#
# #Create Block matching object.
# stereo = cv.StereoSGBM_create(minDisparity= min_disp,
#  numDisparities = num_disp,
#  blockSize = 5,
#  uniquenessRatio = 5,
#  speckleWindowSize = 5,
#  speckleRange = 5,
#  disp12MaxDiff = 1,
#  P1 = 8*3*win_size**2,
#  P2 =32*3*win_size**2)

# #Compute disparity map
# print ("\nComputing the disparity  map...")
# disparity_map = stereo.compute(im1, im2).astype(np.float32) / 16.0
# plt.imshow(disparity_map,'gray')
# plt.show()
#
# disparity_SGBM = stereo.compute(im1, im2)
# wls = cv.ximgproc_DisparityWLSFilter(stereo)
# filtered_disparity_map = wls.filter(disparity_SGBM, im1)
#
# plt.imshow(filtered_disparity_map,'gray')
# plt.show()

# sift = cv.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(images[0][0],None)
# kp2, des2 = sift.detectAndCompute(images[0][1],None)
# # FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)
# flann = cv.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(des1,des2,k=2)
# pts1 = []
# pts2 = []
# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.8*n.distance:
#         pts2.append(kp2[m.trainIdx].pt)
#         pts1.append(kp1[m.queryIdx].pt)
#
# pts1 = np.int32(pts1)
# pts2 = np.int32(pts2)
# F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
# # We select only inlier points
# pts1 = pts1[mask.ravel()==1]
# pts2 = pts2[mask.ravel()==1]


# def drawlines(img1,img2,lines,pts1,pts2):
#     ''' img1 - image on which we draw the epilines for the points in img2
#         lines - corresponding epilines '''
#     r,c = img1.shape
#     # img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
#     # img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
#     for r,pt1,pt2 in zip(lines,pts1,pts2):
#         color = tuple(np.random.randint(0,255,3).tolist())
#         x0,y0 = map(int, [0, -r[2]/r[1] ])
#         x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
#         img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
#         img1 = cv.circle(img1,tuple(pt1),5,color,-1)
#         img2 = cv.circle(img2,tuple(pt2),5,color,-1)
#     return img1,img2

# # Find epilines corresponding to points in right image (second image) and
# # drawing its lines on left image
# lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
# lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(images[0][0],images[0][1],lines1,pts1,pts2)
# # Find epilines corresponding to points in left image (first image) and
# # drawing its lines on right image
# lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
# lines2 = lines2.reshape(-1,3)
# img3,img4 = drawlines(images[0][1],images[0][0],lines2,pts2,pts1)
# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()

def sgbm(rimg1, rimg2):
    # run SGM stereo matching with weighted least squares filtering
    print('Running SGBM stereo matcher...')
    # rimg1 = cv.cvtColor(rimg1, cv.COLOR_BGR2GRAY)
    # rimg2 = cv.cvtColor(rimg2, cv.COLOR_BGR2GRAY)
    maxd = 1
    print('MAXD = ', maxd)
    window_size = 5
    left_matcher = cv.StereoSGBM_create(
        minDisparity=-maxd,
        numDisparities=maxd * 2,
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    lmbda = 8000
    sigma = 1.5
    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(rimg1, rimg2)
    dispr = right_matcher.compute(rimg2, rimg1)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    disparity = wls_filter.filter(displ, rimg1, None, dispr) / 16.0
    return disparity
plt.imshow(sgbm(im1, im2), 'gray')
plt.show()
