import cv2 
 
# 创建一个视频捕获对象，在本例中我们是从文件中读取视频
#vid_capture = cv2.VideoCapture('openCV\\test.mp4')
# vid_capture = cv2.VideoCapture('openCV\\test%02d.png')
vid_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
 
if (vid_capture.isOpened() == False):
  print("Error opening the video file")
# 读取fps和帧数
else:
  # 获取帧率信息
  # 也可以用CAP_PROP_FPS替换5，它们是枚举
  fps = vid_capture.get(5)
  print('Frames per second : ', fps,'FPS')
 
  # 得到帧数
  # 你也可以用CAP_PROP_FRAME_COUNT替换7，它们是枚举
  frame_count = vid_capture.get(7)
  print('Frame count : ', frame_count)
 

###########读视频文件
while(vid_capture.isOpened()):
  # Vid_capture.read()方法返回一个元组，第一个元素是bool类型
  # 第二个是框架
  ret, frame = vid_capture.read()
  if ret == True:
    cv2.imshow('Frame',frame)
    # 20是以毫秒为单位的，试着增加这个值，比如50，然后观察
    key = cv2.waitKey(20)
     
    if key == ord('q'):
      break
  else:
    break

 
# 释放视频捕获对象
vid_capture.release()
cv2.destroyAllWindows()