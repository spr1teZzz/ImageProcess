import cv2 as cv
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
# For webcam input:
cap = cv.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # #如果加载视频，使用“break”代替“continue”
      continue

    #为了提高性能，可选地将图像标记为不可写，以便通过引用传递。
    image.flags.writeable = False
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = holistic.process(image)

    #测试F
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        # image = cv.flip(image, 1)#cv.flip(src,dst,flipCode)flipCode:0---垂直方向翻转,1:水平方向翻转,-1:水平、垂直方向同时翻转
        rgb_frame = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        results2 = face_mesh.process(rgb_frame)#face_mesh 处理图像
        if results2.multi_face_landmarks:
            #print(results.multi_face_landmarks[0].landmark)
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results2.multi_face_landmarks[0].landmark])
            # print(mesh_points.shape)
            # cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            # cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv.circle(image, center_left, int(l_radius), (255,0,255), 1, cv.LINE_AA)
            cv.circle(image, center_right, int(r_radius), (255,0,255), 1, cv.LINE_AA)
            print(center_left,center_right)
    #在图像上绘制地标注释。
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    #将图像水平翻转以显示自拍照视图。
    cv.imshow('MediaPipe Holistic', cv.flip(image, 1))
    key = cv.waitKey(1)
    if key == 27:  # ESC
        break
cap.release()