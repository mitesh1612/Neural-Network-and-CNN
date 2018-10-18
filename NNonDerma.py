import numpy as np
import csv
import random

def kFoldValidation(data,k):
	dataCopy = list(data)
	splitSize = len(data)//k
	testData = []
	# random.seed(0)
	for i in xrange(k):
		temp = []
		for j in xrange(splitSize):
			index = random.randrange(0,len(dataCopy))
			temp.append(dataCopy.pop(index))
		testData.append(temp)
	return testData

def encodeLabels(Y,labels):
	oneHot = np.zeros((Y.shape[0],labels))
	for i in xrange(Y.shape[0]):
		oneHot[i][Y[i]-1] = 1.0
	return oneHot

# def sigmoid(x):
# 	return (1.0/(1.0+np.exp(-x)))
def sigmoid(x):
	return .5 * (1 + np.tanh(.5 * x))

def tanhDerivative(x):
	return 1.0 - np.tanh(x) ** 2

def sigmoidDerivative(x):
	return (sigmoid(x)*(1-sigmoid(x)))

def softmax(outputArray):
	expScores = np.exp(outputArray)
	outputProbs = expScores / np.sum(expScores,axis=1,keepdims=True)
	return outputProbs

def feedforward(eita,X,wh,wout,bh,bout,activationFunction):
	hiddenLayerInput = np.dot(X,wh)
	if activationFunction.lower() == 'tanh':
		hiddenLayerActivations = np.tanh(hiddenLayerInput + bh)
	elif activationFunction.lower() == 'sigmoid':
		hiddenLayerActivations = sigmoid(hiddenLayerInput + bh)
	outputLayerInput = np.dot(hiddenLayerActivations,wout) + bout
	outputProbs = softmax(outputLayerInput)
	return hiddenLayerActivations,outputProbs

def crossEntropyLoss(output,Y):
	index = np.argmax(Y,axis=1).astype(int)
	predProb = output[np.arange(len(output)),index]
	logPreds = np.log(predProb)
	loss = -1.0 * float(np.sum(logPreds))/len(logPreds)
	return loss

def regularizationSoftmaxLoss(regLambda,wh,wout):
	w1loss = 0.5 * regLambda * np.sum(wh*wh)
	w2loss = 0.5 * regLambda * np.sum(wout*wout)
	wloss = w1loss + w2loss
	return wloss 

def backPropagation(X,Y,hiddenLayerActivations,output,wh,bh,wout,bout,eita,activationFunction,regLambda):
	EoutLayer = (output - Y)/output.shape[0]
	delWOut = np.dot(hiddenLayerActivations.T,EoutLayer)
	delBOut = np.sum(EoutLayer,axis=0,keepdims=True)
	if activationFunction == 'tanh':
		tanPrime = tanhDerivative(hiddenLayerActivations)
		EhiddenLayer = np.multiply(tanPrime,EoutLayer.dot(wout.T))
	elif activationFunction == 'sigmoid':
		sigPrime = sigmoidDerivative(hiddenLayerActivations)
		EhiddenLayer = np.multiply(sigPrime,EoutLayer.dot(wout.T))
	delWH = np.dot(X.T,EhiddenLayer)
	delBH = np.sum(EhiddenLayer,axis=0,keepdims=True)
	wout -= (eita*delWOut)
	bout -= (eita*delBOut)
	wh -= (eita*delWH)
	bh -= (eita*delBH)

def checkAccuracy(predictions,labels):
	preds = np.argmax(predictions,1) == np.argmax(labels,1)
	correct = np.sum(preds)
	accuracy = 100.0 * float(correct)/predictions.shape[0]
	return accuracy

original_data = []
filename = 'dermatology.csv'
f = open(filename,'r')
reader = csv.reader(f)
for row in reader:
	if '?' in row:
		continue
	else:
		row = map(int,row)
		if row[-1] not in [1,2,3]:
			continue
		data = row[:-1]
		original_data.append([data,row[-1]])

inputLayerNeurons = len(np.array(original_data[0][0]))
print "Number of Examples: ",len(original_data)
NH = range(4,30,2)
outputLayerNeurons = 3
activationFunction = raw_input("Select your Activation Function.\ntanh (or) sigmoid: ")
np.random.seed(0)
foldValue = 5
if activationFunction.lower() == 'tanh':
	eita = 0.01
elif activationFunction.lower() == 'sigmoid':
	eita = 0.05
else:
	print "Wrong Activation Function Entered!"
	exit()
labels = 3
regLambda = 0.01
result = []
epochs = 1000
print "Number of Epochs:",epochs
for nH in NH:
	foldScores = []
	hiddenLayerNeurons = nH
	wh=np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
	bh=np.random.uniform(size=(1,hiddenLayerNeurons))
	wout=np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
	bout=np.random.uniform(size=(1,outputLayerNeurons))
	print "Number of Hidden Neurons:",nH
	test_data = kFoldValidation(original_data,foldValue)
	for test in test_data:
		train_data = list(test_data)
		train_data.remove(test)
		X = []
		Y = []
		for row in train_data:
			for x,y in row:
				X.append(x)
				Y.append(y)
		X = np.array(X,dtype=float)
		Y = np.array(Y,dtype=int)
		Y = encodeLabels(Y,labels)
		for _ in xrange(epochs):
			hiddenLayerActivations, output = feedforward(eita,X,wh,wout,bh,bout,activationFunction)
			loss = crossEntropyLoss(output,Y)
			loss += regularizationSoftmaxLoss(regLambda,wh,wout)
			backPropagation(X,Y,hiddenLayerActivations,output,wh,bh,wout,bout,eita,activationFunction,regLambda)
			# if _ % 100 == 0:
			# 	print "Loss at Step:",_,"is:",loss
		testX = []
		testY = []
		for row in test:
			testX.append(row[0])
			testY.append(row[1])
		testX = np.array(testX,dtype=float)
		testY = np.array(testY,dtype=int)
		testY = encodeLabels(testY,labels)
		inputLayer = np.dot(testX,wh)
		if activationFunction.lower() == 'sigmoid':
			hiddenLayer = sigmoid(inputLayer)
		elif activationFunction.lower() == 'tanh':
			hiddenLayer = np.tanh(inputLayer)
		hiddenLayer = hiddenLayer + bh
		scores = np.dot(hiddenLayer,wout) + bout
		probs = softmax(scores)
		acc = checkAccuracy(probs,testY)
		foldScores.append(acc)
	print "Validation Scores:"
	print "------------------------------------------"
	print " ".join(map(str,foldScores))
	print "------------------------------------------"
	meanAcc = sum(foldScores)/foldValue
	print "      Accuracy: "+str(meanAcc)+" %."
	result.append(meanAcc)

print "For comparision: "
for x,y in zip(NH,result):
	print "Hidden Units: ",x,"Accuracy: ",y
