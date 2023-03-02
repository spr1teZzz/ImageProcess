import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import utils, math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# 左眼指标
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# 右眼坐标
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

#身体
# BODY = [11,13,21,15,17,19,12,14,16,18,20,23,24]

cap = cv.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera image.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    image.flags.writeable = False
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = holistic.process(image)
	#画图
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    img_h, img_w = image.shape[:2]


    face_mesh = results.face_landmarks
    if face_mesh:
        #打印眼睛的坐标
        # for item in LEFT_EYE:
        #     print(item," :",face_mesh.landmark[item])
        for item in face_mesh.landmark:
             print(item)
    # cv.imshow('MediaPipe Holistic', cv.flip(image, 1))
    # pose_mesh = results.pose_landmarks
    # if pose_mesh:
    #   #打印身体坐标
    #     for item in range(len(pose_mesh.landmark)):
    #         print(item," body:",pose_mesh.landmark[item])
    # cv.imshow('MediaPipe Holistic', image)
    if cv.waitKey(5) & 0xFF == 27: 
      break
cap.release()

