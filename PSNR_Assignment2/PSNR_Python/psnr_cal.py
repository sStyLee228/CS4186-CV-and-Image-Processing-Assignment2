import math
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

def psnr(img1, img2):
    # img1 = img1[:, :, 0]*0.299 + img1[:, :, 1]*0.587 + img1[:, :, 2]*0.114
    # img1 = img1[:, :, 0]*0.299 + img1[:, :, 1]*0.587 + img2[:, :, 2]*0.114

    mse = np.mean( ((img1 - img2)) ** 2 )
    if mse == 0:
        return 'INF'
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def sgbm(rimg1, rimg2, path):
    # run SGM stereo matching with weighted least squares filtering
    print('Running SGBM stereo matcher...')
    # rimg1 = cv.cvtColor(rimg1, cv.COLOR_BGR2GRAY)
    # rimg2 = cv.cvtColor(rimg2, cv.COLOR_BGR2GRAY)
    maxd = 1
    window_size = 5
    left_matcher = cv.StereoSGBM_create(
        minDisparity=-maxd,
        numDisparities=64,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=5,
        speckleRange=32,
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

# When calculate the PSNR:
# 1.) The pixels in ground-truth disparity map with '0' value will be neglected.
# 2.) The left part region (1-250 columns) of view1 is not included as there is no
#   corresponding pixels in the view5.
        [h,l] = gt_img.shape
        gt_img = gt_img[:, 250:l]
        pred_img = pred_img[:, 250:l]
        pred_img[gt_img==0]= 0

        peaksnr = psnr(pred_img,gt_img);
        print('The Peak-SNR value is %0.4f \n', peaksnr);



if __name__== '__main__':
    test()
