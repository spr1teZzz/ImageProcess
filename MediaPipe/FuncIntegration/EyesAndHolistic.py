import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
# 变量
frame_counter =0
CEF_COUNTER =0
TOTAL_BLINKS =0
# 常量
CLOSED_EYES_FRAME =3
FONTS =cv.FONT_HERSHEY_COMPLEX

# 面部边界指数
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# 嘴唇坐标指数
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# 左眼指标
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# 右眼坐标
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

map_face_mesh = mp.solutions.face_mesh
# 摄像头对象
camera = cv.VideoCapture(0)
# 坐标检测函数
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]
    # 返回每个坐标的元组列表 
    return mesh_coord

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

# 眼睛提取函数
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # 将彩色图像转换为比例图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # 获取图像的尺寸
    dim = gray.shape

    # 从灰度暗淡创建蒙版
    mask = np.zeros(dim, dtype=np.uint8)

    # 在白色的mask上绘制眼睛形状
    #cv2.fillPoly()函数可以用来填充任意形状的图型.可以用来绘制多边形,
    #工作中也经常使用非常多个边来近似的画一条曲线.cv2.fillPoly()函数可以一次填充多个图型.
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # 展示mask
    # cv.imshow('mask', mask)
    
    # 在蒙版上绘制眼睛图像，在那里白色的形状
    #cv2.bitwise_and(iamge,image,mask=mask)1 RGB图像选取掩膜选定的区域 2 求两张图片的交集
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # 除眼睛外，将黑色改为灰色
    # cv.imshow('eyes draw', eyes)
    eyes[mask==0]=155
    
    # 得到右眼和左眼的最小和最大x和y值
    # 对于右眼
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # 对于左眼
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # 从mask上裁剪下眼睛
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # 返回剪裁过的眼睛
    return cropped_right, cropped_left

# 眼睛位置估计器
def positionEstimator(cropped_eye):
    # 测量眼睛的高度和宽度
    h, w =cropped_eye.shape
    
    # 去除图像中的噪声
    gaussain_blur = cv.GaussianBlur(cropped_eye, (9,9),0)
    median_blur = cv.medianBlur(gaussain_blur, 3)

    # 应用阈值来转换binary_image
    ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

    # 创建眼睛固定部分
    piece = int(w/3) 

    # 把眼睛分割成三部分
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece+piece]
    left_piece = threshed_eye[0:h, piece +piece:w]
    
    # 调用像素计数器函数
    eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

    return eye_position, color 

# 创建像素计数器函数
def pixelCounter(first_piece, second_piece, third_piece):
    # 计算每个部分的黑色像素
    right_part = np.sum(first_piece==0)
    center_part = np.sum(second_piece==0)
    left_part = np.sum(third_piece==0)
    # 创建这些值的列表
    eye_parts = [right_part, center_part, left_part]

    # 获取列表中Max值的索引
    max_index = eye_parts.index(max(eye_parts))
    pos_eye ='' 
    if max_index==0:
        pos_eye="RIGHT"
        color=[utils.BLACK, utils.GREEN]
    elif max_index==1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index ==2:
        pos_eye = 'LEFT'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye="Closed"
        color = [utils.GRAY, utils.YELLOW]
    return pos_eye, color


with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
        # For webcam input:
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        # 开始时间在这里
        start_time = time.time()
        # 开始视频循环。
        while True:
            frame_counter +=1 # 帧计数器
            ret, frame = camera.read() # 从相机获取帧
            if not ret: 
                break # 不再有帧 break
            #  调整框架
            
            frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
            frame_height, frame_width= frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            #face_mesh处理 result:被检测的人脸集合,其中每个人脸表示为468个人脸坐标的列表，
            results  = face_mesh.process(rgb_frame) #每个坐标由x, y和z组成，x和y分别由图像宽度和高度归一化。
            if results.multi_face_landmarks:
                # 坐标检测函数
                mesh_coords = landmarksDetection(frame, results, False)
                # 眼睛的比率
                ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
                utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)

                if ratio >3.5: #眨眼的宽高比
                    CEF_COUNTER +=1
                    # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                    utils.colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )

                else:
                    if CEF_COUNTER>CLOSED_EYES_FRAME:
                        TOTAL_BLINKS +=1
                        CEF_COUNTER =0
                # cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)
                utils.colorBackgroundText(frame,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)
                
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)

                # 眨眼检测器计数器完成
                right_coords = [mesh_coords[p] for p in RIGHT_EYE]
                left_coords = [mesh_coords[p] for p in LEFT_EYE]
                #眼睛提取函数
                crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
                # cv.imshow('right', crop_right)
                # cv.imshow('left', crop_left)
                #眼睛位置估计器
                eye_position, color = positionEstimator(crop_right)
                utils.colorBackgroundText(frame, f'R: {eye_position}', FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)
                eye_position_left, color = positionEstimator(crop_left)
                utils.colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8, 8)

                #为了提高性能，可选地将图像标记为不可写，以便通过引用传递。
                frame.flags.writeable = False
                # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                results = holistic.process(frame)

                #在图像上绘制地标注释。
                frame.flags.writeable = True
                #frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    frame,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style())
                        


            # 计算帧/秒FPS
            end_time = time.time()-start_time
            fps = frame_counter/end_time

            frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
            # 缩略图图形的图像
            # cv.imwrite(f'img/frame_{frame_counter}.png', frame)
            cv.imshow('frame', frame)
            key = cv.waitKey(1)
            if key == 27:  # ESC
                break
        cv.destroyAllWindows()
        camera.release()