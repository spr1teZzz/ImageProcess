import cv2
import numpy as np
 
img = cv2.imread('openCV\\test.png')
print(img.shape) # 打印 image shape
cv2.imshow("original", img)
 
# 裁剪图像
cropped_image = img[80:280, 150:330]
 
# 展示裁剪图像
cv2.imshow("cropped", cropped_image)
 
# 保存裁剪图像
cv2.imwrite("Cropped test.png", cropped_image)
 
cv2.waitKey(0)
cv2.destroyAllWindows()