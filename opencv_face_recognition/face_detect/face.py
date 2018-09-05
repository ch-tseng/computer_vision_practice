import cv2
import numpy
import imutils

img_width_resize = 1024
cascade_path='../xmls/haarcascade_frontalface_default.xml'
scaleFactor = 1.2
minNeighbors = 10
minFaceSize = (30,30)

cascade = cv2.CascadeClassifier(cascade_path)


def face_detect(img):
    img = imutils.resize(img, width=img_width_resize)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor= scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minFaceSize,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    i = 0
    for (x,y,w,h) in faces:
        roi_color = img[y:y+h, x:x+w]
        cv2.imshow("Face #{}".format(i), roi_color)
        cv2.imwrite("output/face"+str(i)+".jpg", roi_color)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        i += 1

    return img

im = cv2.imread('../demo_images/faces1.jpg')
imgDetected = face_detect(im)

cv2.imshow("Detect", imgDetected)
cv2.imwrite("output/detected.jpg", imgDetected)

cv2.waitKey(0)
