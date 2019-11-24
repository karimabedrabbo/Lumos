import cv2
import numpy as np

from PIL import Image
import glob

import color_utils
import std_dev_calc
import color_thresholding
import color_range
import find_disk


input_extension = ".jpg"
output_extension = ".jpg"
folder_name = "new"


path = "backup/not_glaucoma/*" + input_extension
name_constant = 0

def compute(input_image, name_constant):
	try:

		# #
		# #
		# # STEP ONE: Supply the image and get ROI / other info
		# #
		# #

		roi = input_image

		# find standard deviation
		std_dev = std_dev_calc.std(roi)


		# #
		# #
		# # STEP THREE: Find Prominent Colors
		# #
		# #
		cup_above, cup_below, disk_above, disk_below, color_array = color_range.find_range(roi, std_dev)


		# #
		# #
		# # STEP FOUR: Color thresholding
		# #
		# #

		dilated_disk, dilated_cup, disk_threshed, cup_threshed = color_thresholding.thresh(roi, cup_above, cup_below, disk_above, disk_below)

		# #
		# #
		# # STEP FIVE: Calculations / Analysis
		# #
		# #


		# detect contours in the images, and find the largest one
		disk_contours, _ = cv2.findContours(dilated_disk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		disk_contours = sorted(disk_contours, key = cv2.contourArea, reverse = True)[:5]


		#
		#
		# FOR DISK
		#
		#

		# Scan the cup contours for largest item
		array_sizes = [len(numpoints) for numpoints in disk_contours]


		largest = max(array_sizes)
		index_of_object = array_sizes.index(largest)

		# Largest detected contour
		disk_cnt = disk_contours[index_of_object] 


		# Find the largest and smallest y value
		y_values = [points[0][1] for points in disk_cnt]


		largest_y = max(y_values)
		smallest_y = min(y_values)

		vertical_disk = largest_y - smallest_y


		# Find the largest and smallest x value
		x_values = [points[0][0] for points in disk_cnt]

		largest_x = max(x_values)
		smallest_x = min(x_values)

		horizontal_disk = largest_x - smallest_x

		# disk metric
		disk_avg =  ((1.0 * horizontal_disk) + vertical_disk) / 2


		crop_constant = 15

		new_cropped = roi[int(smallest_y - crop_constant):int(largest_y + crop_constant), int(smallest_x - crop_constant):int(largest_x + crop_constant)]

		name = folder_name + "/" + str(name_constant) + output_extension
		cv2.imwrite(name, new_cropped)

		# display the final image
		# cv2.drawContours(disk_threshed, [disk_cnt], -1, (128,255,0), 3)
		# final = Image.fromarray(disk_threshed).show()
		pass

	except ValueError:
		name = folder_name + "/" + str(name_constant) + output_extension
		cv2.imwrite(name, input_image)
		pass
		


for fname in glob.glob(path):
	img = cv2.imread(fname)
	
	compute(img, name_constant)

	name_constant = name_constant + 1










