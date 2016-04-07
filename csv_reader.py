from Utils import Util
from Calculator import *
import math
import random
import operator
import numpy as np

class LearningStrategy:
	def __init__(self, algorithm=None):
		if(algorithm):
			self.algorithm = algorithm()

	def learn(self,data):
		if(self.algorithm):
			return self.algorithm.learn(data)

class Greedy:
	def learn(self,data):
		return data

class kNN:
	def learn(self,data):
		self.processData(data,0.67)
		
		######### TEST processData FUNCTION ###########################
		#print(repr(len(self.trainingData)))
		#print(repr(len(self.testingData)))
		
		########## TEST euclideanDistance FUNCTION ####################
		#distance = self.euclideanDistance(self.trainingData[0],self.trainingData[1],32)
		#print("Distance: " + repr(distance))

		########### TEST findNearestNeighbors FUNCTION #################
		#trainSet = [[2,2,2,'a'],[4,4,4,'b']]
		#testInstance = [5,5,5]
		#neighbors = self.findNearestNeighbors(trainSet,testInstance,1,1)
		neighbors = self.findNearestNeighbors(self.trainingData,self.testingData[0],0,5) #the last parameter should always be odd to avoid tie votes when it comes to prediction
		#print(self.testingData[0])
		print(neighbors)

		########## TEST getPrediction FUNCTION #########################
		prediction = self.getPrediction(neighbors)
		print(prediction)

	def processData(self,data,split):
		self.trainingData = np.empty((0,33),int)
		self.testingData = np.empty((0,32),int)
		#self.outputData = np.empty((0,len(data)),int)

		#split the data randomly into training and test datasets
		for element in data:
			if random.random() < split:
				self.trainingData = np.append(self.trainingData, np.array([Util.interpretData(element)]), axis=0)
			else:
				out = int(element.pop("G3",None)) #remove the column we want to predict from the test dataset
				self.testingData = np.append(self.testingData, np.array([Util.interpretData(element)]), axis=0)
				#self.outputData = np.append(self.outputData,out)

		#normalize training and test data
		#self.trainingData = self.trainingData/self.trainingData.max(axis=0)
		self.testingData = self.testingData/self.testingData.max(axis=0)

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
			response = neighbors[x][-1] #get the last item in the list (in our case the value for G3)
			sumOfG3 += response

		prediction = sumOfG3/len(neighbors)
		return prediction

class NeuralNetwork:
	def learn(self,data):
		self.processData(data)
		np.random.seed(1)
		connection = (2*np.random.random((len(self.inputData[0]),1))) - 1
		#print(connection)
		
		for i in range(1):
			layer0 = self.inputData
			layer1 = self.nonLinear( np.dot(layer0,connection) )
			layer1_errors = self.outputData - layer1
			layer1_delta = layer1_errors * self.nonLinear(layer1,True)
			connection += np.dot(layer0.T, layer1_delta)
			#print(str(np.mean(np.abs(layer1_errors))))
		print(layer1)

	def processData(self,data):
		self.inputData = np.empty((0,32),int)
		self.outputData = np.empty((0,len(data)),int)
		for element in data:
			out = int(element.pop("G3",None))
			self.inputData = np.append(self.inputData, np.array([Util.interpretData(element)]), axis=0)
			self.outputData = np.append(self.outputData,out)

		self.outputData = np.array([self.outputData]).T

		#normalize input output
		self.inputData = self.inputData/self.inputData.max(axis=0)
		self.outputData = self.outputData/self.outputData.max(axis=0)

		'''self.inputData = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
								   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
								   [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
		self.outputData = np.array([[1,0,1]]).T'''

	def nonLinear(self,output,derivate=False):
		#print(output)
		'''if derivate == True:
			return output*(1-output)
		return 1/(1+np.exp(-output))'''
		if derivate == True:
			return (1-np.power(np.tanh(output),2))
		return np.tanh(output)
		'''if derivate == True:
			return -2*output*np.exp(np.power(-output,2))
		return np.exp(np.power(-output,2))'''

class Genetics:
	def learn(self,data):
		return data

	class _Population:
		def __init__(self,chromosomes):
			self.individuals = []
			for element in data:
				individual = _Individual(element)
				self.individuals.append(individual)

		def getIndividual(self,index):
			return self.individuals[index]

		def getFittess(self):
			fittess = self.individuals[0]
			for element in self.individuals:
				if fittess.getFitness <= element.getFitness:
					fittess = element

			return fittess

		def size(self):
			return len(self.individuals)

		def saveIndividual(self,index,individual):
			self.individuals[index] = individual

	class _Individual:
		def __init__(self,chromosome):
			self.fitness = 0
			self.genes = chromosome

		def getGene(self, index):
			return self.genes[index]

		def size(self):
			return len(self.genes)

		def getFitness(self):
			pass

class BayesNaive:
	def learn(self,data):
		self.processData(data,33)

	def processData(self,data,attributeCount):
		dataset = np.empty((0,attributeCount),int)
		
		for element in data:	
			#making sure G3 is the last element
			out =  int(element.pop("G3",None))
			temp = Util.interpretData(element)
			temp.append(out)
			
			dataset = np.append(dataset, [np.array(temp)], axis=0)

		trainSet, testSet = Util.splitDataset(dataset, 0.75)
		model = self.summarizeClass(trainSet)
		results = self.getPredictions(model,testSet)
		confidence = self.getAccuracy(testSet,results)
		print(results)
		print(confidence)

	def separateByClass(self,dataset):
		classes = {}
		for data in dataset:
			if(data[-1] not in classes):
				classes[data[-1]] = []
			classes[data[-1]].append(data)
		return classes

	def summarize(self,dataset):
		mean = Calculator(formula=Mean)
		stdev = Calculator(formula=STdev)
		result = [(mean.calculate(data), stdev.calculate(data)) for data in zip(*dataset)]
		del result[-1]
		return result

	def summarizeClass(self,dataset):
		classes = self.separateByClass(dataset)
		results = {}
		for index, data in classes.items():
			results[index] = self.summarize(data)
		return results

	def classesProbabilities(self,summaries, input):
		probs = {}
		gaus = Calculator(formula=Gaussian)
		for index,values in summaries.items():
			probs[index] = 1
			for i in range(len(values)):
				mean, stdev = values[i]
				x = input[i]
				probs[index] *= gaus.calculate([x, mean, stdev])
		return probs

	def predict(self,summaries, inputs):
		probs = self.classesProbabilities(summaries,inputs)
		bestMatch = None
		matchProb = -1
		for index, value in probs.items():
			if bestMatch is None or value > matchProb:
				matchProb = value
				bestMatch = index
		return bestMatch

	def getPredictions(self,summaries,inputs):
		results = []
		for item in inputs:
			result = self.predict(summaries,item)
			results.append(result)
		return results

	def getAccuracy(self,test,predictions):
		correct = 0
		for index in range(len(test)):
			if test[index][-1] == predictions[index]:
				correct += 1
		return (correct/len(test))*100


def main():
	data = Util.readInterpretData('learning_dataset.csv')
	algo = LearningStrategy(algorithm=BayesNaive)
	proc_data = algo.learn(data)

if __name__ == "__main__":main()