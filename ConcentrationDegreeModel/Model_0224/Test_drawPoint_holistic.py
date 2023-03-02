import cv2 as cv
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
mp_holistic = mp.solutions.holistic

pointIndex = [13,11,12,14,23,24]

cap = cv.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    image=cv.flip(image, 1)
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
    img_h,img_w,img_c = image.shape
    mlist = []
    for item in pointIndex:
        mlist.append([int(results.pose_landmarks.landmark[item].x * img_w) ,int(results.pose_landmarks.landmark[item].y * img_h),results.pose_landmarks.landmark[item].z])
    # print(results.pose_landmarks.landmark[11])
    narray = np.copy(mlist)
    # print(type(narray)) #<class 'numpy.ndarray'>
    x = mlist[0][0]
    y = mlist[0][1]
    for i in range(len(mlist)):
        if i !=0:
           image = cv.line(image, (x, y), (mlist[i][0],mlist[i][1]), (0,255,0),3) 
           x = mlist[i][0]
           y = mlist[i][1]
    cv.imshow('MediaPipe Holistic', image) 
    if cv.waitKey(5) & 0xFF == 27: 
        break
cap.release()