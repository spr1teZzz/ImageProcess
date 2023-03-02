import cv2
import mediapipe as mp
from time import time as t
import tensorflow as tf
import numpy as np
from pynput.keyboard import Key, Controller
mp_face_mesh = mp.solutions.face_mesh
keyboard = Controller()
MODEL = 'ConcentrationDegreeModel\\Model_0301\\models\\model_pose.hdf5'
timeWindowSize = 30
THRESHOLD =0.5
MAXPREDICTIONSWINDOW = 10
# 左眼指数
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# 右眼指数
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
# 指数表
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

allPointIndex=[6,474,475, 476, 477,469, 470, 471, 472]
level = ['中','左','右']
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
            poseList_tmp = []
            img_h,img_w,img_c = image.shape
            if results.multi_face_landmarks:
                for item in allPointIndex:
                    poseList_tmp.append([int(results.multi_face_landmarks[0].landmark[item].x * img_w),int(results.multi_face_landmarks[0].landmark[item].y * img_h)])
                #将数据与第一个鼻子坐标相关联
                poseList = []
                for it in poseList_tmp:
                    poseList.append([it[0]-poseList_tmp[0][0],it[1]-poseList_tmp[0][1]])   
                #删除第一个点
                poseList=poseList[1:]

                #展开形成 [tmpVector[0][0],tmpVector[0][1],tmpVector[1][0],tmpVector[1][1],...]
                poseList = [item for sublist in poseList for item in sublist]
                
                #归一化
                maxValue = max(poseList, key=abs)
                poseList = [item/maxValue for item in poseList]
                #预测
                prediction = model(np.array([poseList]))[0]
                # print(prediction)
                index_max = int(tf.argmax(prediction))
                prediction = [index_max,float(prediction[index_max])]
                #打印
                # print(prediction)
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
            cv2.imshow('MediaPipe Iris', image)          
            key = cv2.waitKey(5)
            if key & 0xFF == 27: # If ESC is pressed, exit
                break
    cap.release()
if __name__ == '__main__':
    main()