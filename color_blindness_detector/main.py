import cv2 as cv

image = cv.imread('images\colorful-birds.jpg')
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

mode = cv.RETR_EXTERNAL
font = cv.FONT_HERSHEY_SIMPLEX
method = cv.CHAIN_APPROX_SIMPLE

LOWER_BLUE = (71, 107, 78)
HIGHER_BLUE = (117, 255, 211)
blue_mask = cv.inRange(hsv, LOWER_BLUE, HIGHER_BLUE)
blue_contours = cv.findContours(blue_mask, mode, method)[0]
blue_contours = sorted(blue_contours, key=cv.contourArea, reverse=True)[:2]

if blue_contours:
    for contour in blue_contours:
        (cx, cy), radius = cv.minEnclosingCircle(contour)
        cv.putText(image, "blue", (int(cx), int(cy)), font, 0.9, (0, 0, 0), 1)
        cv.circle(image, (int(cx), int(cy)), int(radius), (0, 0, 0), 3)

LOWER_RED = (0, 50, 50)
HIGHER_RED = (10, 255, 255)
red_mask = cv.inRange(hsv, LOWER_RED, HIGHER_RED)
red_contours = cv.findContours(red_mask, mode, method)[0]
red_contours = sorted(red_contours, key=cv.contourArea, reverse=True)[:2]

if red_contours:
    for contour in red_contours:
        (cx, cy), radius = cv.minEnclosingCircle(contour)
        cv.putText(image, "red", (int(cx), int(cy)), font, 0.9, (0, 0, 0), 1)
        cv.circle(image, (int(cx), int(cy)), int(radius), (0, 0, 0), 3)

cv.imshow("Color Detected", image)
cv.waitKey(0)
