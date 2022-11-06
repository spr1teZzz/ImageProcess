import cv2
img = cv2.imread('openCV\sample.jpg')
# 显示图象
cv2.imshow('Original Image',img)
cv2.waitKey(0)
# 如果图像为空，打印错误消息
if img is None:
    print('Could not read image')

# 画直线
imageLine = img.copy()
#从A点画到B点
pointA = (200,80)
pointB = (450,80)
cv2.line(imageLine, pointA, pointB, (255, 255, 0), thickness=3, lineType=cv2.LINE_AA)
cv2.imshow('Image Line', imageLine)
cv2.waitKey(0)

#画圆
# 复制图像
imageCircle = img.copy()
# 确定圆心
circle_center = (415,190)
# 定义圆的半径
radius =100
#  使用circle()函数绘制一个圆
cv2.circle(imageCircle, circle_center, radius, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA) 
# 显示结果
cv2.imshow("Image Circle",imageCircle)
cv2.waitKey(0)

#画实心圆
# 复制图像
imageFilledCircle = img.copy()
# 定义圆心 
circle_center = (415,190)
# 定义圆的半径
radius =100
# 在输入图像上绘制填充圆
cv2.circle(imageFilledCircle, circle_center, radius, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
# 显示输出图像
cv2.imshow('Image with Filled Circle',imageFilledCircle)
cv2.waitKey(0)


#画矩形
# 复制图像
imageRectangle = img.copy()
# 定义矩形的起点和终点
start_point =(300,115)
end_point =(475,225)
# 画矩形
cv2.rectangle(imageRectangle, start_point, end_point, (0, 0, 255), thickness= 3, lineType=cv2.LINE_8) 
#显示输出
cv2.imshow('imageRectangle', imageRectangle)
cv2.waitKey(0)

#画椭圆
# 复制图像
imageEllipse = img.copy()
#定义椭圆中心点
ellipse_center = (415,190)
#定义椭圆的长轴和短轴
axis1 = (100,50)
axis2 = (125,50)
#水平
cv2.ellipse(imageEllipse, ellipse_center, axis1, 0, 0, 360, (255, 0, 0), thickness=3)
#垂直
cv2.ellipse(imageEllipse, ellipse_center, axis2, 90, 0, 360, (0, 0, 255), thickness=3)
#显示输出
cv2.imshow('ellipse Image',imageEllipse)
cv2.waitKey(0)

#画半椭圆
#复制图像
halfEllipse = img.copy()
#定义半椭圆的中心
ellipse_center = (415,190)
#定义轴点
axis1 = (100,50)
#绘制不完全椭圆
cv2.ellipse(halfEllipse, ellipse_center, axis1, 0, 180, 360, (255, 0, 0), thickness=3)
# 填充椭圆
cv2.ellipse(halfEllipse, ellipse_center, axis1, 0, 0, 180, (0, 0, 255), thickness=-2)
#显示输出
cv2.imshow('halfEllipse',halfEllipse)
cv2.waitKey(0)


#复制图像
imageText = img.copy()
#图像上的文本
text = 'I am a Happy dog!'
#org:文本的位置
org = (50,350)
#在输入图像上写入文本
cv2.putText(imageText, text, org, fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1.5, color = (250,225,100))
#显示带有文本的输出图像
cv2.imshow("Image Text",imageText)
cv2.waitKey(0)

cv2.destroyAllWindows()