import cv2
import numpy as np;
 
# Read image
im = cv2.imread("openCV\\blob.jpg", cv2.IMREAD_GRAYSCALE)

#设置默认参数的检测器。
detector = cv2.SimpleBlobDetector_create()
 
#检测斑点
keypoints = detector.detect(im)
 
#将检测到的斑点绘制为红色圆圈。
# cv2。DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS确保圆的大小对应于blob的大小
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# 显示关键点
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)


