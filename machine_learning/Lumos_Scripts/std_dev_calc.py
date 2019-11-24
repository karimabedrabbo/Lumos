import cv2

def std(roi):
	gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	(means, std) = cv2.meanStdDev(gray_roi)
	std_dev = std[0][0]

	return std_dev