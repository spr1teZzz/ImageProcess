import cv2
import mediapipe as mp
from time import time as t
import tensorflow as tf
import numpy as np
import os
from pynput.keyboard import Key, Controller

import tensorflow as tf
import glob

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
mp_face_mesh = mp.solutions.face_mesh
keyboard = Controller()
# level = ['anger','contempt','disgust','fear','happy','sadness','surprise']
# level_name = ['生气','厌恶','害怕','高兴','难过','正常','惊讶']
emotions = ['anger','disgust','fear','happy','sad','surprised','normal']


# emotions = {
#     0: ['Angry', (0,0,255), (255,255,255)],
#     1: ['Disgust', (0,102,0), (255,255,255)],
#     2: ['Fear', (255,255,153), (0,51,51)],
#     3: ['Happy', (153,0,153), (255,255,255)],
#     4: ['Sad', (255,0,0), (255,255,255)],
#     5: ['Surprise', (0,255,0), (255,255,255)],
#     6: ['Neutral', (160,160,160), (255,255,255)]
# }
num_classes = len(emotions)
input_shape = (48, 48, 1)
weights_1 = 'ConcentrationDegreeModel\\Model_0330\\saved_models\\vggnet.h5'
weights_2 = 'ConcentrationDegreeModel\\Model_0330\\saved_models\\vggnet_up.h5'
weights_3 = 'ConcentrationDegreeModel\\Model_0330\\saved_models\\fer_2013.h5'
class VGGNet(Sequential):
    def __init__(self, input_shape, num_classes, checkpoint_path, lr=1e-3):
        super().__init__()
        self.add(Rescaling(1./255, input_shape=input_shape))
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.5))

        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.4))

        self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.5))

        self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.4))

        self.add(Flatten())
        
        self.add(Dense(1024, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(256, activation='relu'))

        self.add(Dense(num_classes, activation='softmax'))

        self.compile(optimizer=Adam(learning_rate=lr),
                    loss=categorical_crossentropy,
                    metrics=['accuracy'])
        
        self.checkpoint_path = checkpoint_path

model_1 = VGGNet(input_shape, num_classes, weights_1)
model_1.load_weights(model_1.checkpoint_path)

model_2 = VGGNet(input_shape, num_classes, weights_2)
model_2.load_weights(model_2.checkpoint_path)

model_3 = VGGNet(input_shape,num_classes,weights_3)
model_3.load_weights(model_3.checkpoint_path)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)


def detection_preprocessing(image, h_max=360):
    h, w, _ = image.shape
    if h > h_max:
        ratio = h_max / h
        w_ = int(w * ratio)
        image = cv2.resize(image, (w_,h_max))
    return image

def resize_face(face):
    x = tf.expand_dims(tf.convert_to_tensor(face), axis=2)
    return tf.image.resize(x, (48,48))

def recognition_preprocessing(faces):
    x = tf.convert_to_tensor([resize_face(f) for f in faces])
    return x

def read_image(read_path,file_pathname,num,accuracy_list):
    #遍历该目录下的所有图片文件
    accuracyNum = 0
    total = 0
    invalid = 0
    #加载模型
    for filename in os.listdir(read_path+'\\'+file_pathname):
        total+=1
        print(filename)
        fpath=read_path+'\\'+file_pathname+'\\'+filename
        image=cv2.imread(fpath)
        image = detection_preprocessing(image)
        H, W, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)
        if results.detections:
            faces = []
            pos = []
            if results.detections:
                detection = results.detections
                box = detection[0].location_data.relative_bounding_box
                x = int(box.xmin * W)
                y = int(box.ymin * H)
                w = int(box.width * W)
                h = int(box.height * H)
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(x + w, W)
                y2 = min(y + h, H)
                face = image[y1:y2,x1:x2]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                faces.append(face)
                pos.append((x1, y1, x2, y2))
            
            x = recognition_preprocessing(faces)
            y_3 = model_3.predict(x)
            l = np.argmax(y_3, axis=1)
            if l == num:
               accuracyNum+=1
        else:
             invalid+=1
    accuracy_list.append([accuracyNum,invalid,total])
if __name__ == '__main__':
    read_path = 'ConcentrationDegreeModel\\DataSets\\ExpW_image_align_filtrate' 
    accuracy_list = []
    for i in range(len(emotions)):
        read_image(read_path,emotions[i],i,accuracy_list)
    print(accuracy_list)
    for i in range(len(accuracy_list)):
        print("************************************")
        print(emotions[i]+'准确率:')
        print('正确:'+str(accuracy_list[i][0]))
        print('干扰:'+str(accuracy_list[i][1]))
        print('总数:'+str(accuracy_list[i][2]))
        acc = accuracy_list[i][0]/(accuracy_list[i][2]-accuracy_list[i][1])
        print('正确率:'+str(acc))
        print("************************************")