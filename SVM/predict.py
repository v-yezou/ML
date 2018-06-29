import numpy as np
import cv2
from os.path import dirname, join, basename
import sys
from glob import glob

bin_n = 16 * 16

def hog(img):
	x_pixel, y_pixel = 194, 259
	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
	mag, ang = cv2.cartToPolar(gx, gy)
	bins = np.int32(bin_n * ang / (2 * np.pi))
	bin_cells = bins[:x_pixel/2, :y_pixel/2], bins[x_pixel/2:, :y_pixel/2], bins[:x_pixel/2, y_pixel/2:], bins[x_pixel/2:, y_pixel/2:]
	mag_cells = mag[:x_pixel/2, :y_pixel/2], mag[x_pixel/2:, :y_pixel/2], mag[:x_pixel/2, y_pixel/2:], mag[x_pixel/2:, y_pixel/2:]

	hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
	hist = np.hstack(hists)

	return hist

img = {}
num = 0

svm = cv2.SVM()
svm.load('svm_cat_data.dat')

test_temp = []
for fn in glob(join(dirname(__file__)+'/predict', '*.jpg')):
	img = cv2.imread(fn, 0)
	test_temp.append(img)
print len(test_temp), 'len(test_temp)'

hogdata = map(hog, test_temp)
testData = np.float32(hogdata).reshape(-1, bin_n*4)
print testData.shape, 'testData'
result = [svm.predict(eachone) for eachone in testData]
print result