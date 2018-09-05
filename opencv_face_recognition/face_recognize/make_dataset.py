import os, glob
import cv2
import dlib
import numpy
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils

train_source = "peoples/train"
test_source = "peoples/test"
cascade_path='../xmls/lbpcascade_frontalface.xml'
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
img_width_resize = 1024
minFaceSize = (30,30)
face_output_width = 120
scaleFactor = 1.1
minNeighbors = 6

def alignFace_imutils(image, foldername, filename):
    image = imutils.resize(image, width=img_width_resize)

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
        rect = dlib.rectangle(x,y,x+w,y+h)
        faceAligned = fa.align(image, gray, rect)
        print("    write to face {}".format(filename))
        cv2.imwrite(foldername + filename+ "_" + str(i) + ".jpg" , faceAligned)
        i += 1

for datasetPath in (train_source, test_source):

    for folders in glob.glob(datasetPath+"/*"):
        print("Load {} ...".format(folders))
        label = os.path.basename(folders)
        label_en, label_tw = label.split()

        i = 0
        for filename in os.listdir(folders):  
            if label_en is not None:
                image_name, image_extension = os.path.splitext(filename)
                image_extension = image_extension.lower()

                if(image_extension == ".jpg" or image_extension==".jpeg" or image_extension==".png" or image_extension==".bmp"):
                    image = cv2.imread(folders + "/" + filename)
                    alignFace_imutils(image, datasetPath + "/faces/", label_en+"_"+str(i))
                    i += 1
