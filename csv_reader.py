from Utils import Util
import math

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

class Genetics:
	def learn(self,data):
		return data

	class _Population:
		
		def __init__(self,data):
			pass

		def getIndividual(self,index):
			return self.individual[index]

		def getFittess(self):
			pass

		def size(self):
			pass

		def saveIndividual(self,individual):
			pass

	class _Individual:
		def __init__(self,data):
			pass

		def generateIndividual(self):
			pass

		def setDefaultGeneLength(self):
			pass

		def getGene(self):
			pass

		def size(self):
			pass

		def getFitness(self):
			pass

		def toString(self):
			pass


def main():
	data = Util.readInterpretData('learning_dataset.csv')
	algo = LearningStrategy(algorithm=Genetics)
	proc_data = algo.learn(data)

	trainingSetLimit = math.floor(len(data)*0.8)

	for index in range(len(data)) :
		chromosome = Util.bitConverter(data[index])
		if len(chromosome) == 80 :
			if index <= trainingSetLimit:
				Util.writeToLearningFile("validation.pac", chromosome)
			else:
				Util.writeToLearningFile("learning.pac",chromosome)

if __name__ == "__main__":main()