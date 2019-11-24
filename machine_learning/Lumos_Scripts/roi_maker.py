import cv2
import numpy as np

from PIL import Image
import glob


input_extension = ".png"
output_extension = ".jpg"
folder_name = "glaucoma"


path = folder_name + "/*" + input_extension
name_constant = 1238

for fname in glob.glob(path):
	img = cv2.imread(fname)
	gray = cv2.imread(fname,0)
	im_height, im_width = img.shape[:2]

	b,g,r = cv2.split(img)

	disk = cv2.GaussianBlur(b, (9,9), 5)

	minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(disk)
	brightest_x_disk = maxLoc[0]
	brightest_y_disk = maxLoc[1]

	#figure out how much of the image to crop
	crop_constant = .2 * im_height

	# crop the image around the brightest element
	roi = img[int(brightest_y_disk - crop_constant):int(brightest_y_disk + crop_constant), int(brightest_x_disk - crop_constant):int(brightest_x_disk + crop_constant)]
		

	name = folder_name + "/" + str(name_constant) + output_extension
	cv2.imwrite(name, roi)

	name_constant = name_constant + 1



	# # Otsu's thresholding
# ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# # Otsu's thresholding after Gaussian filtering
# blur = cv2.GaussianBlur(img,(5,5),0)
# ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

