import cv2
 
# 用于存储边框坐标的列表
top_left_corner=[]
bottom_right_corner=[]
 
#函数，该函数将在鼠标输入时调用
def drawRectangle(action, x, y, flags, *userdata):
     #引用全局变量
    global top_left_corner, bottom_right_corner
     #当按下鼠标左键时，标记左上角
    if action == cv2.EVENT_LBUTTONDOWN:
        top_left_corner = [(x,y)]
        #当释放鼠标左键时，标记右下角
    elif action == cv2.EVENT_LBUTTONUP:
        bottom_right_corner = [(x,y)]    
         #绘制矩形
        cv2.rectangle(image, top_left_corner[0], bottom_right_corner[0], (0,255,0),2, 8)
        cv2.imshow("Window",image)
    
#读取图像
image = cv2.imread("openCV\\test.png")
#制作一个临时图像，将有助于清除图纸
temp = image.copy()
#创建命名窗口
cv2.namedWindow("Window")
# 发生鼠标事件时调用的highgui函数
cv2.setMouseCallback("Window", drawRectangle)
 
k=0
#按q键时关闭窗口
while k!=113:
  # Display the image
  cv2.imshow("Window", image)
  k = cv2.waitKey(0)
    #如果按下c键，使用虚拟图像清除窗口
  if (k == 99):
    image= temp.copy()
    cv2.imshow("Window", image)
 
cv2.destroyAllWindows()