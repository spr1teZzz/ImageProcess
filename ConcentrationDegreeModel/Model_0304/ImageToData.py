'''
    读取图片 然后转化成坐标
'''

import cv2
import os
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
level = ['anger','contempt','disgust','fear','happy','sadness','surprise']
read_path = 'ConcentrationDegreeModel\\Model_0304\\Image\\'
write_path = 'ConcentrationDegreeModel\\Model_0304\\training_data\\'
def read_image(file_pathname,num):
    #遍历该目录下的所有图片文件
    for filename in os.listdir(read_path+'\\'+file_pathname):
        print(filename)
        #读取图片文件
        image = cv2.imread(read_path+'\\'+file_pathname+'\\'+filename)
        
        #利用face_mesh提取人脸坐标
        with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
            #处理图片
            results = face_mesh.process(image)
            img_h,img_w,img_c = image.shape
            #list来装坐标
            face_list_tmp =  []
            if results.multi_face_landmarks:
                for item in results.multi_face_landmarks[0].landmark:
                    face_list_tmp.append([int(item.x * img_w),int(item.y * img_h)])
                    #将数据与第一个两眼之间坐标相关联
            face_list = []
            for it in face_list_tmp:
                face_list.append([it[0]-face_list_tmp[0][0],it[1]-face_list_tmp[0][1]])
            #删除第一个点
            face_list=face_list[1:]

            #展开形成 [tmpVector[0][0],tmpVector[0][1],tmpVector[1][0],tmpVector[1][1],...]
            face_list = [item for sublist in face_list for item in sublist]
            
            #归一化
            maxValue = max(face_list, key=abs)
            face_list = [item/maxValue for item in face_list]   

            #写入
            with open(write_path+level[num]+'.txt', 'a') as f:
                print(f'{face_list}\n')
                f.write(f'{face_list}\n')
                f.close()



for i in range(len(level)):
    read_image(level[i],i)


            

