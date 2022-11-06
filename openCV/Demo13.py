import cv2
image = cv2.imread('openCV\\ContourDetection.jpg')
#将图像转换为灰度格式
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#应用二进制阈值
ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
#可视化二值图像
cv2.imshow('Binary image', thresh)
cv2.waitKey(0)
cv2.imwrite('image_thres1.jpg', thresh)
cv2.destroyAllWindows()
#使用cv2.CHAIN_APPROX_NONE检测二值图像的轮廓。
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                      
#在原始图像上绘制轮廓
image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                
#查看结果
cv2.imshow('None approximation', image_copy)
cv2.waitKey(0)
cv2.imwrite('contours_none_image1.jpg', image_copy)
cv2.destroyAllWindows()


#cv2.CHAIN_APPROX_SIMPLE

#使用cv2.ChAIN_APPROX_SIMPLE检测二值图像上的轮廓。
contours1, hierarchy1 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#在' CHAIN_APPROX_SIMPLE '的原始图像上绘制轮廓
image_copy1 = image.copy()
cv2.drawContours(image_copy1, contours1, -1, (0, 255, 0), 2, cv2.LINE_AA)
#查看结果
cv2.imshow('Simple approximation', image_copy1)
cv2.waitKey(0)
cv2.imwrite('contours_simple_image1.jpg', image_copy1)
cv2.destroyAllWindows()