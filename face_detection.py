import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img=cv2.imread("sample_image.jpg")
img = cv2.resize(img, (900, 600))
grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

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