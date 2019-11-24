import cv2
import numpy as np

from PIL import Image

# img0 = cv2.imread('13.png',0)
# sobelx = cv2.Sobel(img0,cv2.CV_64F,1,0,ksize=5)  # x
# sobely = cv2.Sobel(img0,cv2.CV_64F,0,1,ksize=5)  # y
# magnitude = np.sqrt(sobelx**2+sobely**2)

# cv2.imwrite("out.jpg", sobelx)
# cv2.imwrite("out1.jpg", sobely)
# cv2.imwrite("out2.jpg", magnitude)


# img0 = cv2.imread('out.png',0)
# # # Otsu's thresholding
# ret2,th2 = cv2.threshold(img0,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# # Otsu's thresholding after Gaussian filtering
# blur = cv2.GaussianBlur(img0,(9,9),5)
# ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# one = Image.fromarray(th2).show()
# one = Image.fromarray(th3).show()


# histogram eq
# img = cv2.imread('13.png',0)
# equ = cv2.equalizeHist(img)
# res = np.hstack((img,equ)) #stacking images side-by-side
# cv2.imwrite('res.png',res)	



# histogram eq for colored images

img = cv2.imread('13.png')

img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

res = np.hstack((img,img_output))
cv2.imwrite('res.png',res)