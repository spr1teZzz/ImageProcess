import cv2
import numpy as np

#读取ImageNet类名
with open('Demo15\\input\\classification_classes_ILSVRC2012.txt', 'r') as f:
    image_net_names = f.read().split('\n')
#最终类名(只是一个图像的多个ImageNet名称中的第一个单词)
class_names = [name.split(',')[0] for name in image_net_names]

#加载神经网络模型
model = cv2.dnn.readNet(model='Demo15\\input\\DenseNet_121.caffemodel', 
                      config='Demo15\\input\\DenseNet_121.prototxt', 
                      framework='Caffe')

#从磁盘加载映像
image = cv2.imread('Demo15\\input\\tiger.jpg')
# 从图像创建blob
blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.01, size=(224, 224), 
                             mean=(104, 117, 123))
#设置神经网络的输入blob
model.setInput(blob)
#通过模型转发图像blob
outputs = model.forward()

final_outputs = outputs[0]
#使所有输出为1D
final_outputs = final_outputs.reshape(1000, 1)
#获取类标签
label_id = np.argmax(final_outputs)
#将输出分数转换为softmax概率
probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
#得到最终的最高概率
final_prob = np.max(probs) * 100.

#将最大置信度映射到类标签名称
out_name = class_names[label_id]
out_text = f"{out_name}, {final_prob:.3f}"

#将类名文本放在图像的顶部
cv2.putText(image, out_text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
            2)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.imwrite('Demo15\\outputs\\result_image.jpg', image)
