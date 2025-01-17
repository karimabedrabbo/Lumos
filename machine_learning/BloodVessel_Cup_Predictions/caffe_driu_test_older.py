import numpy as np
import sys
import os
caffe_root = '/Users/karimabedrabbo/caffe'
sys.path.insert(0, caffe_root+'/python/')
import caffe
os.chdir(caffe_root+'/DRIU/')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.misc
from PIL import Image
import scipy.io
#import os
import scipy
#import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder', help='folder input')
args = parser.parse_args()
if args.folder is not None:
	# Point to Caffe folder /path/to/caffe
	caffe_root = '/Users/karimabedrabbo/caffe'

	# Choose between 'DRIVE', 'STARE', 'DRIONS', and 'RIMONE'
	model = 'DRIVE'
	database = args.folder

	# Use GPU?
	use_gpu = 0;
	gpu_id = 0;

	"""
	sys.path.insert(0, caffe_root+'/python/')
	import caffe
	os.chdir(caffe_root+'/DRIU/')
	"""
	def imshow_im(im):
	    plt.imshow(im,interpolation='none',cmap=cm.Grays_r)
	    
	net_struct = 'deploy_'+database+'.prototxt'

	data_root = caffe_root+'/DRIU/Images/'+database+'/'
	save_root = caffe_root+'/DRIU/results/'+database+'/'
	if not os.path.exists(save_root):
	    os.makedirs(save_root)
	
	print(args.folder)

	if args.folder.startswith("1"):
		with open(data_root+'test_glaucoma.txt') as f:
			imnames = f.readlines()

	if args.folder.startswith("0"):
		with open(data_root+'test_not_glaucoma.txt') as f:
			imnames = f.readlines()

	test_lst = [data_root+x.strip() for x in imnames]

	if use_gpu:
	    caffe.set_mode_gpu()
	    caffe.set_device(gpu_id)


	# load net
	#net = caffe.Net('./'+net_struct, './DRIU_'+database+'.caffemodel', caffe.TEST)

	net = caffe.Net('/Users/karimabedrabbo/caffe/DRIU/DRIU_' + model + '/deploy_' + model + '.prototxt', '/Users/karimabedrabbo/caffe/DRIU/DRIU_' + model + '/DRIU_' + model + '.caffemodel', caffe.TEST)
		
	for idx in range(0,len(test_lst)):
	    print("Scoring DRIU for image " + imnames[idx][:-1])
	    
	    #Read and preprocess data
	    im = Image.open(test_lst[idx])
	    in_ = np.array(im, dtype=np.float32)
	    in_ = in_[:,:,::-1] #BGR
	    in_ -= np.array((171.0773,98.4333,58.8811)) #Mean substraction
	    in_ = in_.transpose((2,0,1))
	    
	    #Reshape data layer
	    net.blobs['data'].reshape(1, *in_.shape)
	    net.blobs['data'].data[...] = in_
	    
	    #Score the model
	    net.forward()
	    fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
	    
	    #Save the results
	    scipy.misc.imsave(save_root+imnames[idx][:-1], fuse)
