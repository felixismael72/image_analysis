import cv2 as cv

image = cv.imread("images/ultrassom.jpeg", cv.IMREAD_GRAYSCALE)

negativeImage = 255 - image

cv.imshow("Normal", image)
cv.imshow("Negative", negativeImage)
cv.waitKey(0)
