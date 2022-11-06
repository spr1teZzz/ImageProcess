import cv2 
 
# 创建一个视频捕获对象，在本例中我们是从文件中读取视频
vid_capture = cv2.VideoCapture('openCV\\test.mp4')

# 使用get()方法获取帧大小信息
frame_width = int(vid_capture.get(3))
frame_height = int(vid_capture.get(4))
frame_size = (frame_width,frame_height)
fps = 20

# 初始化视频写入器对象
output = cv2.VideoWriter('output_video_from_file.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, frame_size)
while(vid_capture.isOpened()):
    # Vid_capture.read()方法返回一个元组，第一个元素是bool类型 第二个是框架

    ret, frame = vid_capture.read()
    if ret == True:
        # 将框架写入输出文件
        output.write(frame)
    else:
        print('Stream disconnected')
        break
# 释放对象
vid_capture.release()
output.release()