import cv2
import numpy as np
 
# 使用imread函数读取图像
image = cv2.imread('openCV\car.jpg')
cv2.imshow('Original Image', image)
 
# 使用新的宽度和高度缩小图像
down_width = 300
down_height = 200
down_points = (down_width, down_height)
resized_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
 
# 使用新的宽度和高度来放大图像
up_width = 600
up_height = 400
up_points = (up_width, up_height)
resized_up = cv2.resize(image, up_points, interpolation= cv2.INTER_LINEAR)
# 显示图片
cv2.imshow('Resized Down by defining height and width', resized_down)
cv2.imshow('Resized Up image by defining height and width', resized_up)

# 用不同的插值方法将图像缩小0.6倍
scale_up_x = 1.2
scale_up_y = 1.2
scale_down = 0.6
scaled_f_down = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_LINEAR)
scaled_f_up = cv2.resize(image, None, fx= scale_up_x, fy= scale_up_y, interpolation= cv2.INTER_LINEAR)
res_inter_nearest = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_NEAREST)
res_inter_linear = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_LINEAR)
res_inter_area = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_AREA)
# 将图像在横轴上连接以作比较

vertical= np.concatenate((res_inter_nearest, res_inter_linear, res_inter_area), axis = 0)
cv2.imshow('Inter Nearest :: Inter Linear :: Inter Area', vertical)
cv2.waitKey() 

#按任意键关闭窗口
cv2.destroyAllWindows()