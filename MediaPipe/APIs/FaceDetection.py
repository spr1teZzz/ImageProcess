import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For static images:
IMAGE_FILES = ['MediaPipe\\dlrb.jpg','MediaPipe\\jay.jpg','MediaPipe\\lxp.jpg']
with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    #将BGR图像转换为RGB，并使用MediaPipe人脸检测处理。
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    #绘制每一张脸的面部检测。
    if not results.detections:
      continue
    annotated_image = image.copy()
    for detection in results.detections:
      print('Nose tip:')
      print(mp_face_detection.get_key_point(
          detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
      mp_drawing.draw_detection(annotated_image, detection)
    cv2.imwrite('MediaPipe\\tmp\\annotated_image' + str(idx) + '.png', annotated_image)

#用于摄像头输入:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      #如果加载视频，使用“break”代替“continue”
      continue

    #为了提高性能，可选地将图像标记为不可写入
    #通过引用传递。
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

     #在图像上绘制人脸检测注释。
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)
    #将图像水平翻转以显示自拍照视图。
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    key = cv2.waitKey(1)
    if key ==ord('q'):
      break
cap.release()