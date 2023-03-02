import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(static_image_mode=False,  
                            max_num_faces=3,        
                            refine_landmarks=True,  
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)  as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera image.")   
        # img1 = cv2.imread(imgPath)
        _image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h,w,c = _image.shape
        results = face_mesh.process(_image)
        landmarks = results.multi_face_landmarks
        mlist = []
        for p in landmarks[0].landmark:
            mlist.append([int(p.x * w) ,int(p.y * h),p.z])
        narray = np.copy(mlist)
        # print(type(narray)) #<class 'numpy.ndarray'>
        for p in narray:
            imgCopy = cv2.circle(_image, (int(p[0]), int(p[1])), radius=1, color=(0, 0, 255), thickness=1)
        # plt.imsh ow(_image)
        # plt.show()
        image = cv2.cvtColor(imgCopy, cv2.COLOR_RGB2BGR)
        cv2.imshow('MediaPipe Holistic', image)
        if cv2.waitKey(5) & 0xFF == 27: 
            break
cap.release()