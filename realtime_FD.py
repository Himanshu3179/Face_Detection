import cv2
from random import randrange

from cv2 import waitKey
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
webcam=cv2.VideoCapture(2)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    _,frame=webcam.read()
    grayimg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayimg,scaleFactor = 1.03,minNeighbors=10)
    # print(face_coordinates)
    # x,y,w,h = face_coordinates[0][0],face_coordinates[0][1],face_coordinates[0][2],face_coordinates[0][3]
    if(len(face_coordinates)==0):
        text="NO Faces Detected"
        cv2.putText(frame,text , (50, 50), font, 1, (0, 110, 255), 2, cv2.LINE_4)

    else:
        # print(len(face_coordinates),"Faces Detected")
        text=str(len(face_coordinates))+" Faces Detected"
        cv2.putText(frame,text , (50, 50), font, 1, (10, 25, 100), 2, cv2.LINE_4)
        for i in range(len(face_coordinates)):
            x,y,w,h=face_coordinates[i]

            cv2.rectangle(frame,(x,y) ,(x+w,y+h),(255,0,0),2)
    

    cv2.imshow("output",frame)
    key=cv2.waitKey(1)
    if key==81 or key==113:
        break
webcam.release()
'''
face_coordinates = trained_face_data.detectMultiScale(grayimg)

# print(face_coordinates)
# x,y,w,h = face_coordinates[0][0],face_coordinates[0][1],face_coordinates[0][2],face_coordinates[0][3]
if(len(face_coordinates)==0):
    print("No Face Detected")

    
else:
    print(len(face_coordinates),"Faces Detected")
    for i in range(len(face_coordinates)):
        x,y,w,h=face_coordinates[i]    
        cv2.rectangle(img,(x,y) ,(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)

cv2.imshow("output",img)
# print(len(face_coordinates))

cv2.waitKey()
'''