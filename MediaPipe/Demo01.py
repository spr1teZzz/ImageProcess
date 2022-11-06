import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic #mediapipe 里面 holistic解决方案

file = 'MediaPipe\\dlrb.jpg'
holistic = mp_holistic.Holistic(static_image_mode=True)

image = cv2.imread(file)
image_hight, image_width, _ = image.shape
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = holistic.process(image)

if results.pose_landmarks:
  print(
f'Nose coordinates: ('
f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_hight})'
)
annotated_image = image.copy()
mp_drawing.draw_landmarks(annotated_image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
cv2.imshow('annotated_image',annotated_image)
cv2.imwrite('MediaPipe\\exmaple.jpg', annotated_image)
cv2.waitKey(0)
holistic.close()