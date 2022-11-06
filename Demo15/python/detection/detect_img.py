import cv2
import numpy as np

#加载COCO类名
with open('Demo15\\input\\object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

#为每个类获取不同颜色的数组
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# load the DNN model
model = cv2.dnn.readNet(model='Demo15\\input\\frozen_inference_graph.pb',
                        config='Demo15\\input\\ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                        framework='TensorFlow')

#加载DNN模型
image = cv2.imread('Demo15\\input\\image_2.jpg')
image_height, image_width, _ = image.shape
# 从图像创建blob
blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), 
                             swapRB=True)
# 从图像创建blob
model.setInput(blob)
#前向通过模型进行检测
output = model.forward()

#循环遍历每个检测
for detection in output[0, 0, :, :]:
    # 提取检测的置信度
    confidence = detection[2]
    #只在检测置信度高于一定的阈值，否则跳过
    if confidence > .4:
        #获取类id
        class_id = detection[1]
        #将类id映射到类
        class_name = class_names[int(class_id)-1]
        color = COLORS[int(class_id)]
        #获取边框坐标
        box_x = detection[3] * image_width
        box_y = detection[4] * image_height
        #获取边框的宽度和高度
        box_width = detection[5] * image_width
        box_height = detection[6] * image_height
        #在每个检测到的物体周围画一个矩形
        cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
        #把FPS文本放在帧的顶部
        cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

cv2.imshow('image', image)
cv2.imwrite('Demo15\\outputs\\image_result.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
