import cv2
import numpy as np

from PIL import Image
import glob


input_extension = ".jpg"
output_extension = ".jpg"
folder_name = "all_cropped"


path = folder_name + "/*" + input_extension
name_constant = 1

for fname in glob.glob(path):
	img = cv2.imread(fname)


	name = folder_name + "/" + str(name_constant) + output_extension
	print name
	cv2.imwrite(name, img)

	name_constant = name_constant + 1
