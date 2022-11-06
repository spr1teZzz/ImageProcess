import cv2
 
# 读取图像
image = cv2.imread('openCV\\car.jpg')
 
# 把高度和宽度除以2得到图像的中心
height, width = image.shape[:2]
# 获取图像的中心坐标，创建二维旋转矩阵
center = (width/2, height/2)
 
# 使用cv2.getRotationMatrix2D()来获得旋转矩阵
rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)
 
# 使用cv2.warpAffine旋转图像
rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
 
cv2.imshow('Original image', image) 
cv2.imshow('Rotated image', rotated_image)
# 无限等待，按键盘上的任意键退出
cv2.waitKey(0)
# 将旋转后的图像保存
cv2.imwrite('rotated_image.jpg', rotated_image)