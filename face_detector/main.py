import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    success, img = cap.read()
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv.imshow('Face detection', img)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyWindow()