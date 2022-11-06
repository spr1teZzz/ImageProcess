import cv2 
import numpy as np
 
# 读取图像
image = cv2.imread('openCV\\car.jpg')
 
# 把高度和宽度除以2得到图像的中心
height, width = image.shape[:2]

# 获取用于转换的tx和ty值
# 指定所选的任何值
tx, ty = width / 4, height / 4
 
# 使用tx和ty创建转换矩阵，它是一个NumPy数组
translation_matrix = np.array([
    [1, 0, tx],
    [0, 1, ty]
], dtype=np.float32)


# 平移
translated_image = cv2.warpAffine(src=image, M=translation_matrix, dsize=(width, height))

#展示原图和翻平移图
cv2.imshow('Translated image', translated_image)
cv2.imshow('Original image', image)
cv2.waitKey(0)
# 保存
cv2.imwrite('translated_image.jpg', translated_image)
