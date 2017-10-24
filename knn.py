from Calculator import *
from itertools import combinations
from learningstrategy import LearningStrategy
import math
import random
import operator
import numpy as np



##################################################################################################
########################################## kNN ###################################################
##################################################################################################
class kNN:
	def learn(self,data,dataEvaluation,heuristicStrategy,attributeCount):
		self.processData(data,0.67,attributeCount)
	#def learn(self,data,dataEvaluation,heuristicStrategy,attributeCount):
	#	model = self.processData(data,attributeCount)
	#	self.evaluate(model,dataEvaluation,attributeCount)
	#	return [self.testSet,self.predictions]

		self.predictions = self.getAllPredictions(self.trainingData,self.testingData)
		self.accuracy = self.getAccuracy(self.testingData, self.predictions)

		self.dataEvaluationPredictions = self.evaluate(dataEvaluation,attributeCount)
		#self.dataEvaluationPredictions = self.getAllPredictions(self.trainingData,dataEvaluation)
		return [self.testingData, self.predictions]

	def evaluate(self,dataEvaluation,attributeCount):
		attributesCount = attributeCount-1
		dataset = np.empty((0,attributesCount),int)
		for element in dataEvaluation:
			temp = Util.interpretData(element)
			dataset = np.append(dataset, [np.array(temp)], axis=0)

		evaluation = self.getAllPredictions(self.trainingData,dataset)
		return evaluation

	def processData(self,data,split,attributeCount):
		datasets = np.empty((0,attributeCount),int)

		#split the data randomly into training and test datasets
		for element in data:
			out = int(element.pop("G3",None)) #remove the column we want to predict from the test dataset
			temp = Util.interpretData(element)
			temp.append(out)

			datasets = np.append(datasets, [np.array(temp)], axis=0)

		self.trainingData,self.testingData = Util.splitDataset(datasets, split)

	def euclideanDistance(self,n1,n2,length):
		distance = 0
		for x in range(length):


			distance += np.power((n1[x] - n2[x]),2)
		return np.square(distance)

	def findNearestNeighbors(self,trainingDataSet,testingData,lastRow,k):
		distances = []
		length = len(testingData)-lastRow

		for x in range(len(trainingDataSet)):
			dist = self.euclideanDistance(testingData, trainingDataSet[x], length)
			distances.append((trainingDataSet[x], dist))

		distances.sort(key=operator.itemgetter(1))
		neighbors = []

		for x in range(k):
			neighbors.append(distances[x][0])
		return neighbors

	def getPrediction(self,neighbors):
		sumOfG3 = 0
		prediction = 0

		for x in range(len(neighbors)):
			response = neighbors[x][-1]
			sumOfG3 += response

		prediction = sumOfG3/len(neighbors)
		return prediction

	def getAllPredictions(self,trainData,testData):
		predictions = []
		for x in range(len(testData)):
			neighbors = self.findNearestNeighbors(trainData,testData[x],1,5)
			#print(neighbors)
			predictions.append(self.getPrediction(neighbors))
		return np.ceil(predictions)

	def getAccuracy(self, testDataSet, predictions):
		accurate = 0
		for x in range(len(testDataSet)):
			if testDataSet[x][-1] == predictions[x]:
				accurate += 1
		return (accurate/float(len(testDataSet))) * 100.0

	def showResult(self):
		print("Confiance " + str(self.accuracy))
		print(self.dataEvaluationPredictions)
