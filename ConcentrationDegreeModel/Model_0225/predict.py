import cv2
import mediapipe as mp
from time import time as t
import tensorflow as tf
import numpy as np
from pynput.keyboard import Key, Controller
mp_holistic = mp.solutions.holistic
keyboard = Controller()
MODEL = 'ConcentrationDegreeModel\\Model_0225\\models\\model_pose.hdf5'
timeWindowSize = 30
THRESHOLD = 0.5
MAXPREDICTIONSWINDOW = 10
pointIndex = [13,11,12,14]
allPointIndex =[0,1,2,3,4,5,6,7,8,9,10,20,18,22,16,14,12,11,13,21,15,19,17,14,23]
level = ['非常专注','一般专注','不专注','极不专注']
def main():
    #加载模型
    model = tf.keras.models.load_model(MODEL)
    #摄像头
    cap = cv2.VideoCapture(0)
    timeWindow = []
    scrollingMenu = False
    lastPredictions = []
    gestureActive = True

    with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
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
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            poseList_tmp = []
            img_h,img_w,img_c = image.shape
            if results.pose_landmarks:

                for item in allPointIndex:
                    poseList_tmp.append([int(results.pose_landmarks.landmark[item].x * img_w),int(results.pose_landmarks.landmark[item].y * img_h)])
                
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
                                if lastPredictions[0] == 1:
                                    if not scrollingMenu:
                                        keyboard.press(Key.alt)
                                        keyboard.press(Key.tab)
                                        keyboard.release(Key.tab)
                                        scrollingMenu=True
                                    else:
                                        keyboard.release(Key.alt)
                                        scrollingMenu=False
                                elif lastPredictions[0]==2 or lastPredictions[0]==3:
                                    keyboard.press(Key.left)
                                    keyboard.release(Key.left)
                                elif lastPredictions[0]==4:
                                    keyboard.press(Key.right)
                                    keyboard.release(Key.right)
                                elif lastPredictions[0]==5:
                                    if scrollingMenu:
                                        keyboard.release(Key.alt)
                                    exit()
                            # Switches when the robot is working
                            if lastPredictions[0] == 7:
                                gestureActive=not gestureActive
                                print(f'[INFO] Switching gesture recognition to {gestureActive}')
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
            cv2.imshow('MediaPipe Hands', image)          
            key = cv2.waitKey(5)
            if key & 0xFF == 27: # If ESC is pressed, exit
                break
    cap.release()
if __name__ == '__main__':
    main()