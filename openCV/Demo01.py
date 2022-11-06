import cv2
 
# 函数cv2.imread()用于读取图像。
img_color = cv2.imread('openCV\\test.png',cv2.IMREAD_COLOR)
img_grayscale = cv2.imread('openCV\\test.png',cv2.IMREAD_GRAYSCALE)
img_unchanged = cv2.imread('openCV\\test.png',cv2.IMREAD_UNCHANGED)

#显示窗口内的图像
cv2.imshow('color image',img_color)  
cv2.imshow('grayscale image',img_grayscale)
cv2.imshow('unchanged image',img_unchanged)
 
#等待按键
cv2.waitKey(0)  
 
#删除所有创建的窗口
cv2.destroyAllwindows() 