from Utils import Util
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
		neighbors = self.findNearestNeighbors(0,1) #the last parameter should always be odd to avoid tie votes when it comes to prediction
		#print(self.testingData[0])
		#print(neighbors)

		########## TEST getPrediction FUNCTION #########################
		#predictions = self.getPrediction(neighbors)
		#print(predictions)

	def processData(self,data,split):
		self.trainingData = np.empty((0,33),int)
		self.testingData = np.empty((0,32),int)
		self.outputData = np.empty((0,len(data)),int)

		#split the data randomly into training and test datasets
		for element in data:
			if random.random() < split:
				self.trainingData = np.append(self.trainingData, np.array([Util.interpretData(element)]), axis=0)
			else:
				out = int(element.pop("G3",None)) #remove the column we want to predict from the test dataset
				self.testingData = np.append(self.testingData, np.array([Util.interpretData(element)]), axis=0)
				self.outputData = np.append(self.outputData,out)

		#normalize training and test data
		self.trainingData = self.trainingData/self.trainingData.max(axis=0)
		self.testingData = self.testingData/self.testingData.max(axis=0)
		self.outputData = self.outputData/self.outputData.max(axis=0)
		#print(self.outputData)

	def euclideanDistance(self,n1,n2,length):
		distance = 0
		for x in range(length):
			distance += np.power((n1[x] - n2[x]),2)
		return np.square(distance)

	def findNearestNeighbors(self,lastRow,k):
		distances = []

		#print(testingData)
		#print(str(len(testingData)))

		for x in range(len(self.testingData)):
			length = len(self.testingData[x])-lastRow
			#print(length)
			for y in range(len(self.trainingData)):
				dist = self.euclideanDistance(self.testingData[x], self.trainingData[y], length)
				distances.append((self.trainingData[y], dist))

		distances.sort(key=operator.itemgetter(1))
		print(distances[0])
		#print(str(len(distances)))

		neighbors = {}

		for x in range(len(self.testingData)):	
			for y in range(k):
				#print('x: ' + str(x) + " y: " + str(y))
				neighbors[x,y] = distances[x][y]
				#neighbors.append(distances[x][0])

		#print(neighbors)
		return neighbors

		#for x in range(len(testingData)):


	def getPrediction(self,neighbors):
		sumOfG3 = 0
		prediction = 0

		for x in range(len(neighbors)):
			response = neighbors[x][-1] #get the last item in the list (in our case the value for G3)
			sumOfG3 += response

		prediction = sumOfG3/len(neighbors)
		return prediction

	def evaluateAccuracy(self,predictions):
		accurate = 0
		for x in range(len(self.testingData)):
			if self.testingData[x][-1] is predictions[x]:
				accurate +=1
		return (correct/float(len(self.testingData)))*100

class NeuralNetwork:
	def learn(self,data):
		self.processData(data)
		np.random.seed(1)
		connection = (2*np.random.random((len(self.inputData[0]),1))) - 1
		#print(connection)
		
		for i in range(10000):
			layer0 = self.inputData
			layer1 = self.nonLinear( np.dot(layer0,connection) )
			layer1_errors = self.outputData - layer1
			layer1_delta = layer1_errors * self.nonLinear(layer1,True)
			connection += np.dot(layer0.T, layer1_delta)
			#print(str(np.mean(np.abs(layer1_errors))))
		#print(layer1)

	def processData(self,data):
		self.inputData = np.empty((0,32),int)
		self.outputData = np.empty((0,len(data)),int)
		for element in data:
			out = int(element.pop("G3",None))
			self.inputData = np.append(self.inputData, np.array([Util.interpretData(element)]), axis=0)
			self.outputData = np.append(self.outputData,out)

		self.outputData = np.array([self.outputData]).T

		#normalize input output
		print(self.inputData.max(axis=0))
		print(self.inputData.min(axis=0))
		self.inputData = self.inputData/self.inputData.max(axis=0)
		self.outputData = self.outputData/self.outputData.max(axis=0)
		print(self.inputData)
		print(self.outputData)
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

def main():
	data = Util.readInterpretData('learning_dataset.csv')
	algo = LearningStrategy(algorithm=kNN)
	proc_data = algo.learn(data)


	'''trainingSetLimit = math.floor(len(data)*0.8)

	for index in range(len(data)):
		print(Util.dataBitConverter(data[index]))

	for index in range(len(data)) :
		chromosome = Util.bitConverter(data[index])
		if len(chromosome) == 80 :
			if index <= trainingSetLimit:
				Util.writeToLearningFile("validation.pac", chromosome)
			else:
				Util.writeToLearningFile("learning.pac",chromosome)'''

if __name__ == "__main__":main()