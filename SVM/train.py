import numpy as np
import cv2
from os.path import dirname,join,basename
import sys
from glob import glob

bin_n = 16 * 16

def hog(img):
    x_pixel, y_pixel = 194, 259
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))
    bin_cells = bins[:x_pixel /2, :y_pixel/2], bins[x_pixel/2:, :y_pixel/2],bins[:x_pixel/2, y_pixel/2:],bins[x_pixel/2:, y_pixel/2:]
    mag_cells = mag[:x_pixel /2, :y_pixel/2], mag[x_pixel/2:, :y_pixel/2],mag[:x_pixel/2, y_pixel/2:],mag[x_pixel/2:, y_pixel/2:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    print hist.shape
    print type(hist)
    return hist

img = {}
num = 0
for fn in glob(join(dirname(__file__)+'/cat', '*.jpg')):
	img[num] = cv2.imread(fn, 0)
	num= num + 1
print num
positive = num
for fn in glob(join(dirname(__file__)+'/other', '*.jpg')):
	img[num] = cv2.imread(fn, 0)
	num = num + 1
print num
print positive, 'positive'

trainpic = []
for i in img:
	trainpic.append(img[i])

svm_params = dict(kernel_type = cv2.SVM_LINEAR,
	svm_type = cv2.SVM_C_SVC,
	C = 2.67, gamma = 5.383)

hogdata = map(hog, trainpic)
print np.float32(hogdata).shape, 'hogdata'
trainData = np.float32(hogdata).reshape(-1, bin_n*4)
print trainData.shape, 'trainData'

responses = np.float32(np.repeat(1.0, trainData.shape[0])[:,np.newaxis])
responses[positive:trainData.shape[0]] = -1;
print responses.shape, 'responses'
print len(trainData)
print len(responses)
print type(trainData)

svm = cv2.SVM();
svm.train(trainData, responses, params = svm_params)
svm.save('svm_cat_data.dat')
