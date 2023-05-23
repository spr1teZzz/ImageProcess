import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import time
import glob

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam


emotions = {
    0: ['Angry', (0,0,255), (255,255,255)],
    1: ['Disgust', (0,102,0), (255,255,255)],
    2: ['Fear', (255,255,153), (0,51,51)],
    3: ['Happy', (153,0,153), (255,255,255)],
    4: ['Sad', (255,0,0), (255,255,255)],
    5: ['Surprise', (0,255,0), (255,255,255)],
    6: ['Neutral', (160,160,160), (255,255,255)]
}
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
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)


def detection_preprocessing(image, h_max=500):
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

def write_to_txt(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(str(item) + '\n')

def inference(image):
    H, W, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)
    if results.detections:
        faces = []
        pos = []
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            # mp_drawing.draw_detection(image, detection)
            # print(box)
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
        print(x.shape)
        y_2 = model_2.predict(x)
        l = np.argmax(y_2,axis=1)
        # print(l)
        for i in range(len(faces)):
            cv2.rectangle(image, (pos[i][0],pos[i][1]),
                            (pos[i][2],pos[i][3]), emotions[l[i]][1], 2, lineType=cv2.LINE_AA)
            
            cv2.rectangle(image, (pos[i][0],pos[i][1]-20),
                            (pos[i][2]+20,pos[i][1]), emotions[l[i]][1], -1, lineType=cv2.LINE_AA)
            
            cv2.putText(image, f'{emotions[l[i]][0]}', (pos[i][0],pos[i][1]-5),
                            0, 0.6, emotions[l[i]][2], 2, lineType=cv2.LINE_AA)
    return image

#image
def infer_single_image(path):
    image = cv2.imread(path)
    image = detection_preprocessing(image)
    result = inference(image)
    cv2.imwrite('ConcentrationDegreeModel\\Model_0330\\run\\inference\\out.jpg', result)

def infer_multi_images(paths): 
    for i, path in enumerate(paths):
        image = cv2.imread(path)
        image = detection_preprocessing(image)
        result = inference(image)
        cv2.imwrite('ConcentrationDegreeModel\\Model_0330\\run\\inference\\out_'+str(i)+'.jpg', result)

# infer_single_image('ConcentrationDegreeModel\\Model_0330\\images\\fourPeople2.png')
out = cv2.imread('ConcentrationDegreeModel\\Model_0403\\image\\test02.png')
cv2.imshow('predict',out)
cv2.waitKey(0) #等待按键


# if __name__ == '__main__':
#     cap = cv2.VideoCapture(0)
#     while cap.isOpened():
#         #读取摄像头
#         success, image = cap.read()
#         if not success: 
#             print('[INFO] Ignoring empty camera frame.')
#             continue
#         image.flags.writeable = False
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = detection_preprocessing(image)
#         result = inference(image)
#         image = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
#         cv2.imshow('predict',image)
#         key = cv2.waitKey(5)
#         if key & 0xFF == 27: # If ESC is pressed, exit
#             break





