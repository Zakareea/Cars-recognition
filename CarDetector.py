import cv2 as cv
import time

video = cv.VideoCapture("cars.mp4")
video.set(3, 800), video.set(4, 600)

car_cascade = cv.CascadeClassifier("cas4.xml")

while video.isOpened():

	ret, frame = video.read()
	frame = cv.resize(frame, (600, 400))
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	cars = car_cascade.detectMultiScale(gray, 1.3, 5)
	for (x, y, w, h) in cars:
		cv.rectangle(frame, (x, y - 10), (x + w, y + h), (0, 0, 255), 2)

	cv.imshow("cars", frame)

	key = cv.waitKey(1)
	if key & 0xFF == 27:
		break

	time.sleep(0.01)

video.release()
cv.destroyAllWindows()