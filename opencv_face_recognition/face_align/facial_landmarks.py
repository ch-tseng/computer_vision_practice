import cv2
import dlib
import numpy
import imutils

output_folder = "output/"
img_width_resize = 800
#PREDICTOR_PATH = "../dlib/shape_predictor_68_face_landmarks.dat"
PREDICTOR_PATH = "../dlib/shape_predictor_5_face_landmarks.dat"
cascade_path='../xmls/haarcascade_frontalface_default.xml'

predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade = cv2.CascadeClassifier(cascade_path)

def get_landmarks(im):
    im = imutils.resize(im, width=img_width_resize)
    rects = cascade.detectMultiScale(im, 1.2, 10)
    for (x,y,w,h) in rects:
        rect=dlib.rectangle(x,y,x+w,y+h)
        landmarks = numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

        im = annotate_landmarks(im, landmarks)

    return im

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.8,
                    color=(255, 0, 0))
        cv2.circle(im, pos, 5, color=(0, 255, 0))
    return im

#im=cv2.imread('../demo_images/portrait.jpg')
im=cv2.imread('../demo_images/faces2.jpg')
landmarks = get_landmarks(im)

cv2.imshow("display", landmarks)
cv2.imwrite(output_folder+"landmark.jpg", landmarks)
cv2.waitKey(0)
