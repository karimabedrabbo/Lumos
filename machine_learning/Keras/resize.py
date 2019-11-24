from PIL import Image
import glob

path = "/Users/karimabedrabbo/Desktop/Archive2/dataset3/not_glaucoma/"
img_names = glob.glob1(path,"*.jpg")

img_names.sort()

for item in img_names:
        im = Image.open(path+item)
        imResize = im.resize((128,128), Image.ANTIALIAS)
        imResize.save(path + item , quality=100)
        print item