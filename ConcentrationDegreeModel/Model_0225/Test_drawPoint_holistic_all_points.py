import cv2 as cv
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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

    mp_drawing.draw_landmarks(
    image,
    results.pose_landmarks,
    mp_holistic.POSE_CONNECTIONS,
    landmark_drawing_spec=mp_drawing_styles
    .get_default_pose_landmarks_style())

    cv.imshow('MediaPipe Holistic', image) 
    if cv.waitKey(5) & 0xFF == 27: 
        break
cap.release()