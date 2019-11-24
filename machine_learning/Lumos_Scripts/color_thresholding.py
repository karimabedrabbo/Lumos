import cv2
import numpy as np

def thresh(roi, cup_above, cup_below, disk_above, disk_below):

	# #
	# #
	# # STEP FIVE: Color thresholding
	# #
	# #

	#blur the image first
	blurred_roi = cv2.GaussianBlur(roi, (9,9), 9)

	# remember, openCV uses BGR instead of RGB, but right here we need to use HSV
	# for reference: http://stackoverflow.com/questions/10948589/choosing-correct-hsv-values-for-opencv-thresholding-with-inranges


	# Disk --> threshold for orange colors
	# ORANGE_MIN = np.array([3, 50, 227],np.uint8)
	# ORANGE_MAX = np.array([17, 235, 255],np.uint8)

	ORANGE_MIN = np.array(disk_below,np.uint8)
	ORANGE_MAX = np.array(disk_above,np.uint8)

	hsv_disk = cv2.cvtColor(blurred_roi,cv2.COLOR_BGR2HSV)
	disk_threshed = cv2.inRange(hsv_disk, ORANGE_MIN, ORANGE_MAX)

	# Cup --> threshold for yellow/white colors
	# YELLOW_MIN = np.array([8, 114, 150],np.uint8)
	# YELLOW_MAX = np.array([28, 180, 255],np.uint8)

	YELLOW_MIN = np.array(cup_below,np.uint8)
	YELLOW_MAX = np.array(cup_above,np.uint8)

	hsv_cup = cv2.cvtColor(blurred_roi,cv2.COLOR_BGR2HSV)
	cup_threshed = cv2.inRange(hsv_cup, YELLOW_MIN, YELLOW_MAX)


	# # Veins --> threshold for yellow/white colors

	# RED_MIN = np.array(veins_below,np.uint8)
	# RED_MAX = np.array(veins_above,np.uint8)

	# hsv_veins = cv2.cvtColor(blurred_roi,cv2.COLOR_BGR2HSV)
	# veins_threshed = cv2.inRange(hsv_veins, RED_MIN, RED_MAX)

	#show both
	# display = Image.fromarray(disk_threshed).show()
	# display = Image.fromarray(cup_threshed).show()


	dilated_disk = cv2.dilate(disk_threshed, np.ones((20, 20)))
	dilated_cup = cv2.dilate(cup_threshed, np.ones((20, 20)))
	# dilated_veins = cv2.dilate(veins_threshed, np.ones((20, 20)))

	# display = Image.fromarray(dilated_cup).show()
	# display = Image.fromarray(dilated_disk).show()

	return dilated_disk, dilated_cup, disk_threshed, cup_threshed


