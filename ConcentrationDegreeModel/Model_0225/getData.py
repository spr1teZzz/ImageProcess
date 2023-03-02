import cv2 as cv
import mediapipe as mp
from time import time as t
pointIndex = [13,11,12,14]
allPointIndex =[0,1,2,3,4,5,6,7,8,9,10,20,18,22,16,14,12,11,13,21,15,19,17,14,23]
level = ['非常专注','一般专注','不专注','极不专注']
mp_holistic = mp.solutions.holistic
DATA_FILE = 'ConcentrationDegreeModel\Model_0225\\training_data\\data'
timeWindowSize = 30
cap = cv.VideoCapture(0)
timeWindow = []
i=0
for item in level:
    print(f'{i} : {level[i]}')
    i+=1
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        # 节省获得帧率的时间
        timeWindow.append(t())
        # 如果它比可接受的尺寸大，就剪掉多余的部分
        if len(timeWindow)>timeWindowSize:
            timeWindow=timeWindow[-timeWindowSize:]
        success, image = cap.read()
        if not success:
            print('[INFO] Ignoring empty camera frame.')
            # 如果加载一个视频，用“break”代替“continue”。
            continue
                    # 要提高性能，可选地将映像标记为不可写入
        # 通过引用传递。
        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = holistic.process(image)

        # 在图像上画上手部注释。
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

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

        # 将图像水平翻转，将显示自拍照视图。
        image =  cv.flip(image, 1)
        # Showing the image
        cv.imshow('MediaPipe pose', image)
        #wait and get the key pressed
        key = cv.waitKey(5)
        if key & 0xFF == 27: # If ESC is pressed, exit
            break
        elif key >= ord('0') and key<=ord('3') and len(poseList): # If a number is pressed
            num = int(chr(key))                 # Store the number
            print(f'[INFO] 获取的专注度等级:{level[num]}')
            print()
            with open(DATA_FILE+str(num)+'.txt', 'a') as f:
                print(f'{poseList}\n')
                f.write(f'{poseList}\n')
                f.close()