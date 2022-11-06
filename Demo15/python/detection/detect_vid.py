import cv2
import time
import numpy as np

#加载COCO类名
with open('Demo15\\input\\object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

#为每个类获取不同颜色的数组
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

#加载DNN模型
model = cv2.dnn.readNet(model='Demo15\\input\\frozen_inference_graph.pb',
                        config='Demo15\\input\\ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                        framework='TensorFlow')

#捕捉视频
cap = cv2.VideoCapture('Demo15\\input\\video_1.mp4')
#获取视频帧的宽度和高度，以便正确保存视频
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
#创建“VideoWriter()”对象
out = cv2.VideoWriter('Demo15\\outputs\\video_result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))

#检测视频每帧中的对象
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        image = frame
        image_height, image_width, _ = image.shape
        # 从图像创建blob
        blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), 
                                     swapRB=True)
        # 开始时间计算FPS
        start = time.time()
        model.setInput(blob)
        output = model.forward()        
        #检测后结束时间
        end = time.time()
        #计算当前帧检测的帧数
        fps = 1 / (end-start)
        #循环遍历每个检测
        for detection in output[0, 0, :, :]:
            #提取检测的置信度
            confidence = detection[2]
            #只在检测置信度高于某个阈值时绘制边界框，否则跳过
            if confidence > .4:
                #获取类id
                class_id = detection[1]
                # 将类id映射到类 
                class_name = class_names[int(class_id)-1]
                color = COLORS[int(class_id)]
                # 获取边界框坐标
                box_x = detection[3] * image_width
                box_y = detection[4] * image_height
                # 获取边界框的宽度和高度
                box_width = detection[5] * image_width
                box_height = detection[6] * image_height
                # 在每个检测到的物体周围画一个矩形
                cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
                # 将类名文本放在检测到的对象上
                cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                # 把FPS文本放在帧的顶部
                cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
        
        cv2.imshow('image', image)
        out.write(image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
