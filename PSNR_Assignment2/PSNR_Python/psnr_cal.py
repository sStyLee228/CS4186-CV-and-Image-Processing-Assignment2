import math
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

def psnr(img1, img2):
    mse = np.mean( ((img1 - img2)) ** 2 )
    if mse == 0:
        return 'INF'
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def sgbm(im1, im2, path):
    maxd = 1
    window_size = 7 #5
    left_matcher = cv.StereoSGBM_create(
        minDisparity=1,
        numDisparities=240,
        blockSize=3,
        P1=144, # 8 * 3 * window_size ** 2,
        P2=576, # 32 * 3 * window_size ** 2,
        # disp12MaxDiff=1,
        # uniquenessRatio=5,
        # speckleWindowSize=5,
        # speckleRange=32,
        preFilterCap=63,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    lmbda = 8000
    sigma = 1.2
    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(im1, im2)
    dispr = right_matcher.compute(im2, im1)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    disparity = wls_filter.filter(displ, im1, None, dispr) / 16.0
    # disparity = cv.normalize(
    #     src=disparity,
    #     dst=disparity,
    #     beta=1,
    #     alpha=255,
    #    	norm_type=cv.NORM_MINMAX,
    # 	dtype=cv.CV_8U
    # )
    # disparity = np.uint8(disparity)
    cv.imwrite(os.path.join(path, 'disp1.png'), disparity, [cv.COLOR_BGR2GRAY])

def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def sift(im1, im2):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    index_params = dict(algorithm = 1, trees = 5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pts1, pts2 = [], []

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    return pts1, pts2

def compute_and_show_epipolar(im1, im2):
    pts1, pts2 = sift(im1, im2)
    pts1, pts2 = np.int32(pts1), np.int32(pts2)
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    pts1, pts2 = pts1[mask.ravel()==1], pts2[mask.ravel()==1]

    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F).reshape(-1, 3)
    im_lines1, im_lines2 = drawlines(im1, im2, lines1, pts1, pts2)

    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F).reshape(-1, 3)
    im_lines3, im_lines4 = drawlines(im2, im1, lines2, pts2, pts1)

    plt.subplot(121), plt.imshow(im_lines1)
    plt.subplot(122), plt.imshow(im_lines3)
    plt.show()


def test():
    test_imgs = ["Art", "Dolls", "Reindeer"]
    this_file_path = os.path.dirname(__file__)

    for index in range(3):
        image_folder_path = os.path.join(this_file_path, 'pred', test_imgs[index])
        gt_folder_path = os.path.join(this_file_path, 'gt', test_imgs[index])

        im_l = cv.imread(os.path.join(image_folder_path, 'view1.png'), 0)
        im_r = cv.imread(os.path.join(image_folder_path, 'view5.png'), 0)

        sgbm(im_l, im_r, image_folder_path)

        gt_names = os.path.join(gt_folder_path, 'disp1.png')
        gt_img = np.array(Image.open(gt_names),dtype=float)

        pred_names =  os.path.join(image_folder_path, 'disp1.png')
        pred_img = np.array(Image.open(pred_names),dtype=float)

        [h,l] = gt_img.shape
        gt_img = gt_img[:, 250:l]
        pred_img = pred_img[:, 250:l]
        pred_img[gt_img==0]= 0

        peaksnr = psnr(pred_img,gt_img);
        print('The Peak-SNR value for %s is %0.4f' % (test_imgs[index], peaksnr));

if __name__== '__main__':
    test()
