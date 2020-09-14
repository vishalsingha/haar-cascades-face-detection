import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('C:\\Users\\Vishal Singh\\Desktop\\Deeplearning\\haarcascades\\haarcascade_frontalface_default1.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\Vishal Singh\\Desktop\\Deeplearning\\haarcascades\\haarcascade_eye1.xml')


cam = cv2.VideoCapture(0)

while 1:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
                
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 7)
        # mouth = mouth_cascade.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)

    if cv2.waitKey(30) & 0xff == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()



