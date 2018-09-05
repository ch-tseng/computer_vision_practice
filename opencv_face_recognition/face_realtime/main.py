import cv2

cam_id = 0
cam_resolution = (1024,768)
cascade = "haarcascade_frontalface_default.xml"
face_size = (47, 62)
scaleFactor = 1.3
minNeighbors = 10

camera = cv2.VideoCapture(cam_id)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_resolution[0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_resolution[1])


face_cascade = cv2.CascadeClassifier('../xmls/' + cascade)
while True:
    (grabbed, img) = camera.read() 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor= scaleFactor,
        minNeighbors=minNeighbors,
        minSize=face_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    i = 0
    for (x,y,w,h) in faces:
        roi_color = img[y:y+h, x:x+w]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

    cv2.imshow("Frame", img)
    cv2.waitKey(1)
