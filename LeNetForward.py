# Forward Pass of LeNet (CNN Example)
import sys
import numpy as np
from PIL import Image
if len(sys.argv) < 2:
	print "Enter the path of Image in Command Line Arguments!"
	sys.exit(1)
IMG_PATH = sys.argv[1]
# Some Constants for LeNet
IMG_ROW_SIZE = 32	#Input of LeNet is 32*32 Image
IMG_COL_SIZE = 32
FILTER_ROW_SIZE = 5
FILTER_COL_SIZE = 5
FILTER_DEPTH_L1 = 6
FILTER_DEPTH_L2 = 16
POOLING_ROW_SIZE = 2
POOLING_COL_SIZE = 2
FILTER_DEPTH_FCC = 120
NN_INPUT_SIZE = 120
NN_HIDDEN_UNITS = 84
NN_OUTPUT_SIZE = 10
# Helper Functions
def createFilter(depth):
	filterMatrix = np.random.randn(FILTER_ROW_SIZE,FILTER_COL_SIZE,depth)
	return filterMatrix

def ReLU(x):
	return np.where(x>0,np.where(x<255.0,x,255),0.0)

def convolveOperation(data,dataFilter):
	convRes = np.multiply(data,dataFilter)
	return convRes.sum()

def convolve(data,datafilter):
	filterRow, filterCol, numFilters = datafilter.shape
	dataRow, dataCol, dataChannels = data.shape
	convResultDim = dataRow - filterRow + 1
	convResult = np.zeros((convResultDim,convResultDim,numFilters))
	for filNum in xrange(numFilters):
		for x in xrange(convResultDim):
			for y in xrange(convResultDim):
				convResult[x][y][filNum] = convolveOperation(data[x:x+filterRow,y:y+filterCol,filNum % dataChannels],datafilter[:,:,filNum])
	return convResult

def maxPooling(data,poolRowSize,poolColSize,stride):
	dataRow, dataCol, resLayer = data.shape
	poolRes = np.zeros(((dataRow-poolRowSize)//stride+1,(dataCol-poolColSize)//stride+1,resLayer))
	for l in xrange(resLayer):
		i = 0
		while i < dataRow:
			j = 0
			while j < dataCol:
				poolRes[i/2,j/2,l] = np.max(data[i:i+poolRowSize,j:j+poolColSize,l])
				j += stride
			i += stride
	return poolRes

# ~~~~~~~~~~~~~~~~~~ NEURAL NETWORK ~~~~~~~~~~~~~~~~~~~~~
def createWeightMatrix():
	wih = np.random.randn(NN_INPUT_SIZE,NN_HIDDEN_UNITS)
	who = np.random.randn(NN_HIDDEN_UNITS,NN_OUTPUT_SIZE)
	return wih,who

def sigmoid(x):
	return 1 / (1+np.exp(-x))

def forwardPass(i):
	wih, who = createWeightMatrix()
	netj = np.dot(i,wih)
	yj = sigmoid(netj)
	netk = np.dot(yj,who)
	zk = sigmoid(netk)
	return zk

def softMax(z):
	expScores = np.exp(z)
	probs = expScores/np.sum(expScores)
	return probs

# Main Code Begins
try:
	img = Image.open(IMG_PATH)
	a1 = np.array(img)
	print "Original Dimensions of the Image:",a1.shape
	img = img.resize((IMG_ROW_SIZE,IMG_COL_SIZE),Image.ANTIALIAS)
	imageArray = np.array(img)
	# To save the reduced sized image
	# imageArray.save("reduced-size.png",)
	print "Reduced Dimensions of the Image:",imageArray.shape
except IOError:
	print "Couldn't find the given image. Try again!"
	sys.exit(1)
# LeNet Architecture :
# INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC(Conv) => RELU => FC
# ~~~~~~~~~~~~~~~~ 1st BLOCK ~~~~~~~~~~~~~~~
# First Layer Filter
filterL1 = createFilter(FILTER_DEPTH_L1)
# print "Filter for first Convolution Layer:"
# print filterL1
# Convoluting at First Layer
convResult = convolve(imageArray,filterL1)
print "Dimensions after 1st Convolution:",convResult.shape
# Applying ReLU Activation
# print convResult
reLURes = ReLU(convResult)
# Visualizing Images
img1 = Image.fromarray(reLURes,'RGB')
img1 = img1.resize((312,312))
img1.show()
# Max Pooling on the block
poolResult = maxPooling(reLURes,POOLING_ROW_SIZE,POOLING_COL_SIZE,2)
# Visualizing Images
img1 = Image.fromarray(poolResult,'RGB')
img1 = img1.resize((312,312))
img1.show()
print "Dimensions after 1st Max Pooling:",poolResult.shape
# ~~~~~~~~~~~~~~~~~~~ 2nd Block ~~~~~~~~~~~~~~~~~~~~~
filterL2 = createFilter(FILTER_DEPTH_L2)
# print "Filter for second Convolution Layer:"
# print filterL2
# Convoluting at Second Layer
convResult2 = convolve(poolResult,filterL2)
print "Dimension after 2nd Convolution:",convResult2.shape
# Applying ReLU Activation
# print convResult2
reLURes2 = ReLU(convResult2)
# Visualizing Images
img1 = Image.fromarray(reLURes2,'RGB')
img1 = img1.resize((312,312))
img1.show()
# Max Pooling on the block
poolResult2 = maxPooling(reLURes2,POOLING_ROW_SIZE,POOLING_COL_SIZE,2)
# Visualizing Images
img1 = Image.fromarray(poolResult2,'RGB')
img1 = img1.resize((312,312))
img1.show()
print "Dimension after 2nd Max Pooling:",poolResult2.shape
# ~~~~~~~~~~~~~~~~~ CONVOLUTION AT FIRST FC LAYER ~~~~~~~~~~~~~~~~~
filterL3 = createFilter(FILTER_DEPTH_FCC)
# print "Filter for the FC Convolution:"
# print filterL3
# Convoluting for FC
print "Filter Dimensions for FC Convolution:",filterL3.shape
convResult3 = convolve(poolResult2,filterL3)
print "Dimensions after FC Convolution:",convResult3.shape
# Applying ReLU Activation
# print convResult3
reLURes3 = ReLU(convResult3)
# Visualizing Images
img1 = Image.fromarray(reLURes3,'RGB')
img1 = img1.resize((312,312))
img1.show()
print "Dimensions after applying ReLU:",reLURes3.shape
# Applying Neural Network on the Result
nnOutput = forwardPass(reLURes3)
# print "Output Shape of Neural Network:",nnOutput.shape
# Applying SoftMax on the NN Output
softmaxRes = softMax(nnOutput[0,0,:])
# print "Result of SoftMax:"
# print softmaxRes
print "Class of Image: "
print np.argmax(softmaxRes)