import cv2
import numpy as np
import glob

def resize_and_increase_contrast(img, scale_percent, alpha=1.5, beta=0):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    contrast_img = cv2.convertScaleAbs(resized_img, alpha=alpha, beta=beta)
    equalized = cv2.bitwise_not(cv2.equalizeHist(contrast_img))
    return equalized

img1 = cv2.imread('sar-data/image1.tif', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('sar-data/image2.tif', cv2.IMREAD_GRAYSCALE)

img1 = resize_and_increase_contrast(img1, 5, 2, 0)
img2 = resize_and_increase_contrast(img2, 5, 2, 0)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
height, width = img1.shape[:2]
warped_img2 = cv2.warpPerspective(img2, H, (width, height))

overlay = cv2.addWeighted(img1, 0.5, warped_img2, 0.5, 0)
cv2.imwrite('overlay.tif', overlay)
