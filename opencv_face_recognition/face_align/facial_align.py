import os, glob
import cv2
import dlib
import numpy
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils

output_folder = "output/"
cascade_path='../xmls/haarcascade_frontalface_default.xml'
PREDICTOR_PATH = "../shape_predictor_68_face_landmarks.dat"
img_width_resize = 1024
minFaceSize = (30,30)
face_output_width = 120
scaleFactor = 1.1
minNeighbors = 4

def alignFace_imutils(image):
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    cascade = cv2.CascadeClassifier(cascade_path)

    fa = FaceAligner(predictor, desiredFaceWidth=face_output_width)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor= scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minFaceSize,
        flags=cv2.CASCADE_SCALE_IMAGE
    )


    i = 0
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        rect = dlib.rectangle(x,y,x+w,y+h)
        faceAligned = fa.align(image, gray, rect)
        cv2.imshow("Face #{}".format(i),  image[y:y+h, x:x+w])
        cv2.imshow("Aligned-{}".format(i), faceAligned)
        cv2.imwrite(output_folder + "align_"+str(i)+".jpg" , faceAligned)
        cv2.waitKey(0)
        i += 1

    cv2.imwrite(output_folder +  "faces.jpg", image)

image = cv2.imread("../demo_images/faces3.jpg")
image = imutils.resize(image, width=img_width_resize)
cv2.imshow("Original", image)

alignFace_imutils(image)
