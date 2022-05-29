import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")

while True:
    ret, frame = cap.read()
    flipMe = cv.flip(frame, 1)

    grayScale = cv.cvtColor(flipMe, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayScale, 1.3, 5)
    for (x, y, w, h) in faces:
        faceRec = cv.rectangle(flipMe, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv.putText(faceRec, "Face", (x, y-10), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 255, 0), 2)
        roi_gray = grayScale[y:y+h, x:x+w]
        roi_color = flipMe[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            eyeRec = cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0))
            cv.putText(eyeRec, "Eye", (x, y-10), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 0, 0), 2)
            #cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0))

    cv.imshow('frame', flipMe)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
