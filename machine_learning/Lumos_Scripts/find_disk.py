import cv2

def find_largest_contour(dilated_disk, dilated_veins, disk_threshed, veins_threshed):

	# #
	# #
	# # STEP SIX: Contour detection
	# #
	# #

	# detect contours in the images, and find the largest one

	(disk_contours, _) = cv2.findContours(dilated_disk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	disk_contours = sorted(disk_contours, key = cv2.contourArea, reverse = True)[:5]

	(vein_contours, _) = cv2.findContours(dilated_veins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	vein_contours = sorted(vein_contours, key = cv2.contourArea, reverse = True)[:5]

	#
	#
	# FOR DISK
	#
	#

	# Scan the cup contours for largest item
	array_sizes = []

	for numpoints in disk_contours:
		array_sizes.append(len(numpoints))

	largest = max(array_sizes)
	index_of_object = array_sizes.index(largest)

	# Largest detected contour
	disk_cnt = disk_contours[index_of_object]
	disk_area = cv2.contourArea(disk_cnt)

	#
	#
	# FOR VEINS
	#
	#

	# Scan the cup contours for largest item
	array_sizes = []

	for numpoints in vein_contours:
		array_sizes.append(len(numpoints))

	largest = max(array_sizes)
	index_of_object = array_sizes.index(largest)

	# Largest detected contour
	vein_cnt = vein_contours[index_of_object] 
	vein_area = cv2.contourArea(vein_cnt)


	""" If the vein area is bigger, it means that the vein is actually the disk and the disk is the vein.
		Since the contours were flip-flopped, we need to flip-flop them back :-) """
	if vein_area < disk_area:
		return dilated_disk, disk_threshed
	else:
		return dilated_veins, veins_threshed

