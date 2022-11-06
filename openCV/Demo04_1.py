import cv2
import numpy as np
img = cv2.imread('openCV\\test.png')
image_copy = img.copy() 
imgheight=img.shape[0]
imgwidth=img.shape[1]
#M、N计算为imgheight、imgwidth 1/3 应该切成9张
M = 227
N = 341
x1 = 0
y1 = 0 
for y in range(0, imgheight, M):#M 为步长
    for x in range(0, imgwidth, N):#N 为步长
        if (imgheight - y) < M or (imgwidth - x) < N:
            break          
        y1 = y + M
        x1 = x + N
        # 检查补丁的宽度或高度是否超过图像的宽度或高度
        if x1 >= imgwidth and y1 >= imgheight:
            x1 = imgwidth - 1
            y1 = imgheight - 1
            #剪成MxN大小的小块
            tiles = image_copy[y:y+M, x:x+N]
            #将每个补丁保存到文件目录中
            cv2.imwrite('openCV\saved_patches\\'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
        elif y1 >= imgheight: # 当补丁高度超过图像高度时
            y1 = imgheight - 1
            #剪成MxN大小的小块
            tiles = image_copy[y:y+M, x:x+N]
            #将每个补丁保存到文件目录中
            cv2.imwrite('openCV\saved_patches\\'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
        elif x1 >= imgwidth: # 当补丁宽度超过图像宽度时
            x1 = imgwidth - 1
            #剪成MxN大小的小块
            tiles = image_copy[y:y+M, x:x+N]
            #将每个补丁保存到文件目录中
            cv2.imwrite('openCV\saved_patches\\'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
        else:
            #剪成MxN大小的小块
            tiles = image_copy[y:y+M, x:x+N]
            #将每个补丁保存到文件目录中
            cv2.imwrite('openCV\saved_patches\\'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
#保存完整的图像到文件目录
cv2.imshow("Patched Image",img)
cv2.imwrite("patched.jpg",img)
cv2.waitKey()
cv2.destroyAllWindows()