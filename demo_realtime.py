# demo_realtime.py
# 调用demo中RecognitionRealtime类
import cv2
from demo import RecognitionRealtime


# 手机摄像头
# url = 'http://admin:admin@xxxx/'
# 笔记本摄像头
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
RecognitionRealtime = RecognitionRealtime()
while(True):
    ret, frame = cap.read()
    result_img = RecognitionRealtime.recognition(frame)
    if result_img == []:
        cv2.imshow('Real Time Recognition', frame)
    else:
        cv2.imshow('Real Time Recognition', result_img)
    k = cv2.waitKey(30) & 0xff
    if k==27: # 按Esc退出
        break

cap.release()
cv2.destroyAllWindows()