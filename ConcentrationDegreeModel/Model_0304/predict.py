import cv2
import mediapipe as mp
from time import time as t
import tensorflow as tf
import numpy as np
from pynput.keyboard import Key, Controller
mp_face_mesh = mp.solutions.face_mesh
keyboard = Controller()
# level = ['anger','contempt','disgust','fear','happy','sadness','surprise']
level = ['生气','轻蔑','厌恶','害怕','高兴','难过','惊讶']
timeWindowSize = 30
THRESHOLD =0.7
MAXPREDICTIONSWINDOW = 10
MODEL = 'ConcentrationDegreeModel\\Model_0304\\models\\model_face.hdf5'
def main():
    #加载模型
    model = tf.keras.models.load_model(MODEL)
    #摄像头
    cap = cv2.VideoCapture(0)
    timeWindow = []
    scrollingMenu = False
    lastPredictions = []
    gestureActive = True

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            #保存帧频
            timeWindow.append(t())
            if len(timeWindow)>timeWindowSize:
                timeWindow=timeWindow[-timeWindowSize:]
            #读取摄像头
            success, image = cap.read()
            if not success:
                print('[INFO] Ignoring empty camera frame.')
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            face_list_tmp = []
            img_h,img_w,img_c = image.shape
            if results.multi_face_landmarks:
                for item in results.multi_face_landmarks[0].landmark:
                    face_list_tmp.append([int(item.x * img_w),int(item.y * img_h)])
                #将数据与第一个鼻子坐标相关联
                face_list = []
                for it in face_list_tmp:
                    face_list.append([it[0]-face_list_tmp[0][0],it[1]-face_list_tmp[0][1]])   
                #删除第一个点
                face_list=face_list[1:]

                #展开形成 [tmpVector[0][0],tmpVector[0][1],tmpVector[1][0],tmpVector[1][1],...]
                face_list = [item for sublist in face_list for item in sublist]
                
                #归一化
                maxValue = max(face_list, key=abs)
                face_list = [item/maxValue for item in face_list]
                #预测
                prediction = model(np.array([face_list]))[0]
                # print(prediction)
                index_max = int(tf.argmax(prediction))
                prediction = [index_max,float(prediction[index_max])]
                #打印
                print(prediction)
                if prediction[1]>THRESHOLD:
                    lastPredictions.append(prediction[0])
                    if len(lastPredictions)>MAXPREDICTIONSWINDOW :
                        lastPredictions=lastPredictions[-MAXPREDICTIONSWINDOW : ]
                        # If all elements are equal
                        if lastPredictions.count(lastPredictions[0]) == len(lastPredictions):
                            if gestureActive:
                                print(f'[INFO] identifyed {level[prediction[0]]}')
            # Flip the image horizontally for a selfie-view display.
            image =  cv2.flip(image, 1)
            # Compute and write fps
            l = len(timeWindow)-1
            if l>0:
                avFPS =0
                for i in range(l,0,-1):
                    avFPS += (timeWindow[i]-timeWindow[i-1])
                avFPS = l/avFPS
                # Write FPS
                text = f'{avFPS:.2f}fps'
                cv2.putText(image, text, (15, 25), cv2.FONT_HERSHEY_TRIPLEX, 1, ( 0, 0, 0), 1)
            # Showing the image
            cv2.imshow('MediaPipe face', image)      
            key = cv2.waitKey(5)
            if key & 0xFF == 27: # If ESC is pressed, exit
                break
    cap.release()
if __name__ == '__main__':
    main()