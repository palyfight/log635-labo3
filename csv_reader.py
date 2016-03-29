from Utils import Util
import math
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

class NeuralNetwork:
	def learn(self,data):
		self.processData(data)

	def processData(self,data):
		self.inputData = np.array([])
		self.ouputData = np.array([])
		for element in data:
			out = element.pop("G3",None)
			self.inputData = np.append(element,self.inputData)
			self.ouputData = np.append(out,self.ouputData)

		for index in range(len(self.ouputData)):
			print (self.inputData[index])
			print (self.ouputData[index])

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
	algo = LearningStrategy(algorithm=NeuralNetwork)
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