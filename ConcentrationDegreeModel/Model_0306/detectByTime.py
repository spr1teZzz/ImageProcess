import cv2
import mediapipe as mp
from time import time as t
import tensorflow as tf
import numpy as np
from pynput.keyboard import Key, Controller
import time
import math
from collections import Counter
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh


CEF_COUNTER =0
TOTAL_BLINKS =0
# 常量
CLOSED_EYES_FRAME =3

SAVE_DIR = 'ConcentrationDegreeModel\\Model_0306\\data\\'

MODEL_FACE = 'ConcentrationDegreeModel\\Model_0306\\models\\model_face.hdf5'
MODEL_IRIS = 'ConcentrationDegreeModel\\Model_0306\\models\\model_iris.hdf5'
MODEL_POSE = 'ConcentrationDegreeModel\\Model_0306\\models\\model_pose.hdf5'

IrisPointIndex=[6,474,475, 476, 477,469, 470, 471, 472]
PosePointIndex =[0,1,2,3,4,5,6,7,8,9,10,20,18,22,16,14,12,11,13,21,15,19,17,14,23]
# 左眼指标
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# 右眼坐标
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

timeWindowSize = 30
iris_level = ['中','左','右']
face_level = ['生气','轻蔑','厌恶','害怕','高兴','难过','惊讶']
pose_level = ['非常专注','一般专注','不专注','极不专注']
def processData(list_tmp,model):
    #将数据与第一个鼻子坐标相关联
    res_list = []
    for it in list_tmp:
        res_list.append([it[0]-list_tmp[0][0],it[1]-list_tmp[0][1]])   
    #删除第一个点
    res_list=res_list[1:]

    #展开形成 [tmpVector[0][0],tmpVector[0][1],tmpVector[1][0],tmpVector[1][1],...]
    res_list = [item for sublist in res_list for item in sublist]
    
    #归一化
    maxValue = max(res_list, key=abs)
    res_list = [item/maxValue for item in res_list]
    #预测
    prediction = model(np.array([res_list]))[0]
    # print(prediction)
    index_max = int(tf.argmax(prediction))
    prediction = [index_max,float(prediction[index_max])]
    return prediction[0]

# 欧几里得距离 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance


# 眼睛的比率
def blinkRatio(img, landmarks, right_indices, left_indices):
    # 右眼 
    # 水平线
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # 垂直线
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # 在右眼上画线
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # 左眼 
    # 水平线
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # 垂直线 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    #左眼距离
    rhDistance = euclaideanDistance(rh_right, rh_left)#水平
    rvDistance = euclaideanDistance(rv_top, rv_bottom)#垂直

    #右眼距离
    lhDistance = euclaideanDistance(lh_right, lh_left)#水平
    lvDistance = euclaideanDistance(lv_top, lv_bottom)#垂直

    #左右眼的距离比
    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    #平均比例
    ratio = (reRatio+leRatio)/2
    return ratio 

time_start =0 # 记录开始时间
time_end = 0 #最终时间

face_list = []
pose_list = []
iris_list = []
Blinking_time = 0 #眨眼次数

#导入模型
model_face = tf.keras.models.load_model(MODEL_FACE)
model_iris = tf.keras.models.load_model(MODEL_IRIS)
model_pose = tf.keras.models.load_model(MODEL_POSE)
#作为是否进行数据处理的标志
flag = False
#摄像头
print('*****按1进行检测,按2停止检测*****')
cap = cv2.VideoCapture(0)
timeWindow = []
with mp_holistic.Holistic(
min_detection_confidence=0.5,
min_tracking_confidence=0.5) as holistic:
    with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    ) as face_mesh: 
        while cap.isOpened():
            #保存帧频
            timeWindow.append(t())
            if len(timeWindow)>timeWindowSize:
                timeWindow=timeWindow[-timeWindowSize:]
            #读取摄像头
            success, image = cap.read()
            # Flip the image horizontally for a selfie-view display.
            image =  cv2.flip(image, 1)
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if not success:
                print('[INFO] Ignoring empty camera frame.')
                continue
            if flag:
                #获取数据,进行处理
                # print('开始进行数据采集')
                img_h,img_w,img_c = image.shape
                #肢体
                pose_image = holistic.process(image)
                pose_list_tmp = []
                if pose_image.pose_landmarks:
                    for item in PosePointIndex:
                        pose_list_tmp.append([int(pose_image.pose_landmarks.landmark[item].x * img_w),int(pose_image.pose_landmarks.landmark[item].y * img_h)])
                pose_list.append(processData(pose_list_tmp,model_pose))

                #面部
                face_image = face_mesh.process(image)
                face_list_tmp = []
                if face_image.multi_face_landmarks:
                    for item in face_image.multi_face_landmarks[0].landmark:
                        face_list_tmp.append([int(item.x * img_w),int(item.y * img_h)])
                face_list.append(processData(face_list_tmp,model_face))

                # #虹膜
                iris_list_tmp = []
                if face_image.multi_face_landmarks:
                    for item in IrisPointIndex:
                        iris_list_tmp.append([int(face_image.multi_face_landmarks[0].landmark[item].x * img_w),int(face_image.multi_face_landmarks[0].landmark[item].y * img_h)])
                iris_list.append(processData(iris_list_tmp,model_iris))
                #眨眼次数
                ratio = blinkRatio(image, face_list_tmp, RIGHT_EYE, LEFT_EYE)
                if ratio >3.5: #眨眼的宽高比
                    CEF_COUNTER +=1
                else:
                    if CEF_COUNTER>CLOSED_EYES_FRAME:
                        Blinking_time+=1
                        CEF_COUNTER =0
                        print('眨眼')
            else:
                curtime = time.strftime('%Y%m%d%H%M', time.localtime())
                if len(face_list) != 0:
                    # print('脸部数据开始处理')
                    with open(SAVE_DIR+'face'+curtime+'.txt', 'w') as f:
                        f.write(f'{face_list}\n')
                        f.close()
                    #进行数据处理
                if len(pose_list) != 0:
                    # print('身体数据开始处理')
                    with open(SAVE_DIR+'pose'+curtime+'.txt', 'w') as f:
                        f.write(f'{pose_list}\n')
                        f.close()
                if len(iris_list) !=0:
                    with open(SAVE_DIR+'iris'+curtime+'.txt', 'w') as f:
                        f.write(f'{iris_list}\n')
                        f.close()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            key = cv2.waitKey(5)
            if key & 0xFF == 27: # If ESC is pressed, exit
                break
            elif key == ord('1'): # If a number is pressed
                print('**********开始进行检测**********')
                time_start = time.time()
                flag = True
            elif key == ord('2'):
                print('**********停止检测**********')
                time_end = time.time() 
                flag = False
            cv2.imshow('MediaPipe face', image) 

#统计出现次数
def countNum(list,res_cnt):
    for i in list:
        res_cnt[i]+=1


#统计出现频率
def countRatio(list):
    res_list = []
    list_sum = sum(list)
    for i in list:
        res_list.append(i/list_sum)
    return res_list

if __name__ == '__main__':
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间
    #统计眼部各种表情出现的频率
    iris_cnt=[0,0,0]
    countNum(iris_list,iris_cnt)
    iris_ratio =  countRatio(iris_cnt)
    #统计脸部各种表情出现的频率
    face_cnt = [0,0,0,0,0,0,0]
    countNum(face_list,face_cnt)     
    face_ratio = countRatio(face_cnt)
    #统计四肢各种表情出现的频率
    pose_cnt = [0,0,0,0]
    countNum(pose_list,pose_cnt)   
    pose_ratio = countRatio(pose_cnt)
    print(str(time_start)+'<--->'+str(time_end)+" 时间("+str(time_sum)+")")
    #计算占比
    print('*****脸部表情*****')
    for i in range(len(face_cnt)):
        print(face_level[i]+'出现的次数:'+str(face_cnt[i])+' 占比:'+str(face_ratio[i]))

    print('*****眼部虹膜*****')
    for i in range(len(iris_cnt)):
        print(iris_level[i]+'出现的次数:'+str(iris_cnt[i])+' 占比:'+str(iris_ratio[i]))

    print('*****四肢*****')
    for i in range(len(pose_cnt)):
        print(pose_level[i]+'出现的次数:'+str(pose_cnt[i])+' 占比:'+str(pose_ratio[i]))
    print('*****眨眼次数*****')
    print("次数:"+str(Blinking_time))
