import cv2
import numpy as np
import color_utils
from sklearn.cluster import KMeans

# preload global functions for speed
rgb2hsv = color_utils.rgb2hsv
hsv2above_cup_low = color_utils.hsv2above_cup_low
hsv2below_cup_low = color_utils.hsv2below_cup_low
hsv2above_low = color_utils.hsv2above_low
hsv2below_low = color_utils.hsv2below_low
hsv2above_cup = color_utils.hsv2above_cup
hsv2below_cup = color_utils.hsv2below_cup
hsv2above = color_utils.hsv2above
hsv2below = color_utils.hsv2below
hsv2above_cup_high = color_utils.hsv2above_cup_high
hsv2below_cup_high = color_utils.hsv2below_cup_high
hsv2above_high = color_utils.hsv2above_high
hsv2below_high = color_utils.hsv2below_high


def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
 
	# return the histogram
	return hist

def find_range(roi, std_dev):

	image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
	 
	img_vec = image.reshape((image.shape[0] * image.shape[1], 3))

	# cluster the pixel intensities
	clt = KMeans(n_clusters = 4)
	clt.fit(img_vec)

	#find what percent that each color is used
	per = centroid_histogram(clt)
	rgb = clt.cluster_centers_
	

	# remove the color with the second to largest percentage (bg)
	max_value = max(per)
	max_index = per.tolist().index(max_value)
	rgb_colors = rgb.tolist()
	rgb_colors.pop(max_index)
	percent = per.tolist()
	percent.pop(max_index)

	# max_value = max(rgb_colors[i][2])
	# max_index = per.tolist().index(max_value)

	# get the color with the smallest percentage (cup)
	cup_value = min(percent)
	cup_index = percent.index(cup_value)

	# get the color with the second largest percentage (disk?)
	disk_value = sorted(percent)[-2]
	disk_index = percent.index(disk_value)

	# get the color with the largest percentage (veins?)
	veins_value = max(percent)
	veins_index = percent.index(veins_value)

	hsv_colors = []

	# rgb to hsv
	for color in rgb_colors:
		hsv_col = color_utils.rgb2hsv(color[0],color[1],color[2])
		hsv_colors.append(hsv_col)

	print("hsl:", hsv_colors)
	hsv_colors = sorted(hsv_colors, key=lambda x: x[1])
	print(hsv_colors)

	cup_index = 0
	disk_index = 1
	veins_index = 2


	"""
	Standard Deviation Thresholding:

	A value lower than 25 means that it's a low-contrast image whose colors don't vary much. 
	This means that the HSV range should be lower.

	A value between 25 and 70 means that it's a mid-contrast image whose colors vary enough. 
	This means that the HSV range should be moderate.

	A value higher than 75 means that it's a high-contrast image whose colors vary a LOT. 
	This means that the HSV range should be higher.

	"""
	if std_dev <= 20:
		# contour 1 (probs cup)
		cup_above = hsv2above_cup_low(hsv_colors[cup_index][0],hsv_colors[cup_index][1],hsv_colors[cup_index][2])
		cup_below = hsv2below_cup_low(hsv_colors[cup_index][0],hsv_colors[cup_index][1],hsv_colors[cup_index][2])

		# contour 2 (probs disk)
		disk_above = hsv2above_low(hsv_colors[disk_index][0],hsv_colors[disk_index][1],hsv_colors[disk_index][2])
		disk_below = hsv2below_low(hsv_colors[disk_index][0],hsv_colors[disk_index][1],hsv_colors[disk_index][2])

		# contour 3 (probs veins)
		# veins_above = color_utils.hsv2above_low(hsv_colors[veins_index][0],hsv_colors[veins_index][1],hsv_colors[veins_index][2])
		# veins_below = color_utils.hsv2below_low(hsv_colors[veins_index][0],hsv_colors[veins_index][1],hsv_colors[veins_index][2])

	elif std_dev > 20 and std_dev <= 70:
		# contour 1 (probs cup)
		cup_above = hsv2above_cup(hsv_colors[cup_index][0],hsv_colors[cup_index][1],hsv_colors[cup_index][2])
		cup_below = hsv2below_cup(hsv_colors[cup_index][0],hsv_colors[cup_index][1],hsv_colors[cup_index][2])

		# contour 2 (probs disk)
		disk_above = hsv2above(hsv_colors[disk_index][0],hsv_colors[disk_index][1],hsv_colors[disk_index][2])
		disk_below = hsv2below(hsv_colors[disk_index][0],hsv_colors[disk_index][1],hsv_colors[disk_index][2])

		# contour 3 (probs veins)
		# veins_above = color_utils.hsv2above(hsv_colors[veins_index][0],hsv_colors[veins_index][1],hsv_colors[veins_index][2])
		# veins_below = color_utils.hsv2below(hsv_colors[veins_index][0],hsv_colors[veins_index][1],hsv_colors[veins_index][2])

	else:
		# contour 1 (probs cup)
		cup_above = hsv2above_cup_high(hsv_colors[cup_index][0],hsv_colors[cup_index][1],hsv_colors[cup_index][2])
		cup_below = hsv2below_cup_high(hsv_colors[cup_index][0],hsv_colors[cup_index][1],hsv_colors[cup_index][2])

		# contour 2 (probs disk)
		disk_above = hsv2above_high(hsv_colors[disk_index][0],hsv_colors[disk_index][1],hsv_colors[disk_index][2])
		disk_below = hsv2below_high(hsv_colors[disk_index][0],hsv_colors[disk_index][1],hsv_colors[disk_index][2])

		# contour 3 (probs veins)
		# veins_above = color_utils.hsv2above_high(hsv_colors[veins_index][0],hsv_colors[veins_index][1],hsv_colors[veins_index][2])
		# veins_below = color_utils.hsv2below_high(hsv_colors[veins_index][0],hsv_colors[veins_index][1],hsv_colors[veins_index][2])


	return cup_above, cup_below, disk_above, disk_below, hsv_colors
