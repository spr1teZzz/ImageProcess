import cv2
image = cv2.imread('openCV\\ContourDetection.jpg')
 
# B, G, R通道分割
blue, green, red = cv2.split(image)
 
#使用蓝色通道和无阈值检测轮廓
contours1, hierarchy1 = cv2.findContours(image=blue, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
 
#在原始图像上绘制轮廓
image_contour_blue = image.copy()
cv2.drawContours(image=image_contour_blue, contours=contours1, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
#查看结果
cv2.imshow('Contour detection using blue channels only', image_contour_blue)
cv2.waitKey(0)
#cv2.imwrite('blue_channel.jpg', image_contour_blue)
cv2.destroyAllWindows()
 
#使用绿色通道和无阈值检测轮廓
contours2, hierarchy2 = cv2.findContours(image=green, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
#在原始图像上绘制轮廓
image_contour_green = image.copy()
cv2.drawContours(image=image_contour_green, contours=contours2, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
#查看结果
cv2.imshow('Contour detection using green channels only', image_contour_green)
cv2.waitKey(0)
#cv2.imwrite('green_channel.jpg', image_contour_green)
cv2.destroyAllWindows()
 
#使用红色通道和无阈值检测轮廓
contours3, hierarchy3 = cv2.findContours(image=red, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
#在原始图像上绘制轮廓
image_contour_red = image.copy()
cv2.drawContours(image=image_contour_red, contours=contours3, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
#查看结果
cv2.imshow('Contour detection using red channels only', image_contour_red)
cv2.waitKey(0)
# cv2.imwrite('red_channel.jpg', image_contour_red)
cv2.destroyAllWindows()