from Utils import Util
from Calculator import *
from itertools import combinations
import math
import random
import operator
import numpy as np

class LearningStrategy:
	def __init__(self, algorithm=None):
		if(algorithm):
			self.algorithm = algorithm()

	def learn(self,data, dataEvaluation, heuristicStrategy=None, attributeCount=33):
		if(self.algorithm):
			return self.algorithm.learn(data,dataEvaluation,heuristicStrategy,attributeCount)

	def getAccuracy(self,test,prediction):
		if(self.algorithm):
			return self.algorithm.getAccuracy(test,prediction)

	def showResult(self):
		if(self.algorithm):
			return self.algorithm.showResult()



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

##################################################################################################
################################### GENETIC ALGORITHM ############################################
##################################################################################################
class Genetics:
	def learn(self,data,dataEvaluation,heuristicStrategy,attributeCount):
		self.xmenRate = 0.095
		self.gladiator = 7
		self.genesPool = data[0].keys()
		population = Population(50,data,dataEvaluation,True,heuristicStrategy)
		winner = population.getFittest()
		#population.getFittest()
		
		generationCount = 0
		while(winner.performance() < 60):
			print("Generation " + str(generationCount) + " Confiance " + str(winner.performance()) + " attributes " + str(winner.getGenome()) )
			winner.showResult()
			generationCount +=1
			population = self.nextGeneration(population,data,dataEvaluation,heuristicStrategy,attributeCount)
			winner = population.getFittest()


	def nextGeneration(self,population,data,dataEvaluation,heuristicStrategy,attributeCount):
		nextGen = Population(1+(population.size()*2),data,dataEvaluation,False,heuristicStrategy)
		nextGen.saveIndividuals(0,population.getFittest())
		count = 1
		for index in range(population.size()):
			challenger = self.tournament(population,data,dataEvaluation,heuristicStrategy,attributeCount)
			champion   = self.tournament(population,data,dataEvaluation,heuristicStrategy,attributeCount)
			nextGenChallenger, nextGenChamp = self.crossover(challenger,champion,int(np.floor(challenger.size()/2)),heuristicStrategy)
			nextGen.saveIndividuals(count, nextGenChamp)
			nextGen.saveIndividuals(count+1, nextGenChallenger)
			count += 2

		for index in range(nextGen.size()):
			neo = self.xmen(nextGen.getIndividual(index))
			nextGen.saveIndividuals(index,neo)
		return nextGen

	def tournament(self,population,data,dataEvaluation,heuristicStrategy,attributeCount):
		gladiators = Population(self.gladiator,data,dataEvaluation,False,heuristicStrategy)
		for index in range(self.gladiator):
			chosenOne = random.randint(0,population.size()-1)
			gladiators.saveIndividuals(index,population.getIndividual(chosenOne))
		morpheus = gladiators.getFittest()
		return morpheus

	def crossover(self, individu1, individu2, mergingPoint,heuristicStrategy):
		newGenome = []
		newGenome2 = []
		newGenome.extend(individu1.getGenome()[:mergingPoint])
		newGenome2.extend(individu2.getGenome()[:mergingPoint])
		newGenome.extend(individu2.getGenome()[mergingPoint:len(individu2.getGenome())])
		newGenome2.extend(individu1.getGenome()[mergingPoint:len(individu1.getGenome())])

		newIndividu1 = Individual(newGenome,heuristicStrategy)
		newIndividu2 = Individual(newGenome2,heuristicStrategy)

		return [newIndividu1, newIndividu2]

	def xmen(self, individu):
		np.random.seed(1)
		genes = [gene for gene in self.genesPool if gene not in individu.getGenome()]
		if random.uniform(0,1) <= self.xmenRate:
			randomKey = random.choice(genes)
			individu.addGene(randomKey)
		return individu

class Population:
	def __init__(self, size, data,dataEvaluation, init = False, heuristicStrategy=None):
		self.individuals = np.empty(size,dtype=Individual)
		self.data = data
		self.dataEvaluation = dataEvaluation
		if init == True:
			genomes = self.generateGenomes(data[0].keys(),size)
			for index in range(len(genomes)):
				individual = Individual(genomes[index],heuristicStrategy)
				self.individuals[index] = individual

	def getIndividual(self,index):
		return self.individuals[index]

	def getFittest(self):
		fittest = self.individuals[0]
		for individual in self.individuals:
			fitness = individual.getFitness(self.data,self.dataEvaluation)
			if fitness > fittest.getFitness(self.data,self.dataEvaluation):
				fittest = individual
		return fittest

	def size(self):
		return len(self.individuals)

	def saveIndividuals(self,index,individual):
		self.individuals[index] = individual

	def generateGenomes(self,data,limit):
		data = list(data)
		data.remove("G3")
		combination = list(combinations(data,2))
		#np.random.seed(5)
		while len(combination) > limit:
			rand = random.randint(0,len(combination)-1)
			combination.pop(rand)
		return combination

class Individual:
	def __init__(self, gene, heuristicStrategy=None):
		self.gene = list(gene)
		self.heuristicStrategy = heuristicStrategy

	def filterData(self, data):
		dataset = []
		for row in data:				
			filters = {}
			for index in self.gene:
				if(index not in filters):
					filters[index] = row[index]
			filters["G3"] = row["G3"]
			dataset.append(filters)
		return dataset

	def filterEvaluation(self, data):
		dataset = []
		for row in data:
			filters = {}
			for index in self.gene:
				if(index not in filters and index != 'G3'):
					filters[index] = row[index]
			dataset.append(filters)
		return dataset

	def getFitness(self,data,dataEvaluation):
		dataset = self.filterData(data)
		dataEvaluation = self.filterEvaluation(dataEvaluation)
		test,prediction = self.heuristicStrategy.learn(dataset,dataEvaluation,attributeCount=len(dataset[0].keys()))
		self.fitness = self.heuristicStrategy.getAccuracy(test,prediction)
		return self.fitness

	def performance(self):
		return self.fitness

	def size(self):
		return len(self.gene)

	def setGene(self,value):
		self.gene = value

	def addGene(self,value):
		self.gene.append(value)

	def getGene(self, index):
		return self.gene[index]

	def getGenome(self):
		return self.gene

	def showResult(self):
		self.heuristicStrategy.showResult()

##################################################################################################
####################################### BAYES NAIF ###############################################
##################################################################################################
class BayesNaive:
	def learn(self,data,dataEvaluation,heuristicStrategy,attributeCount):
		model = self.processData(data,attributeCount)
		self.evaluate(model,dataEvaluation,attributeCount)
		return [self.testSet,self.predictions]


	def processData(self,data,attributeCount):
		dataset = np.empty((0,attributeCount),int)
		
		for element in data:	
			#making sure G3 is the last element
			out =  int(element.pop("G3",None))
			temp = Util.interpretData(element)
			temp.append(out)
			
			dataset = np.append(dataset, [np.array(temp)], axis=0)

		self.trainSet, self.testSet = Util.splitDataset(dataset, 0.67)
		model = self.summarizeClass(self.trainSet)
		results = self.getPredictions(model,self.testSet)
		self.predictions = results
		self.confidence = self.getAccuracy(self.testSet,results)
		return model

	def evaluate(self,model,dataEvaluation,attributeCount):
		attributesCount = attributeCount-1
		dataset = np.empty((0,attributesCount),int)
		for element in dataEvaluation:
			temp = Util.interpretData(element)
			dataset = np.append(dataset, [np.array(temp)], axis=0)

		self.evaluation = self.getPredictions(model,dataset)

	def showResult(self):
		print("Confiance " + str(self.confidence))
		print(self.evaluation)

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


##################################################################################################
########################################## MAIN###################################################
##################################################################################################
def main():
	data = Util.readInterpretData('learning_dataset.csv')
	data2 = Util.readInterpretData('learning_dataset.csv')
	evaluation = Util.readInterpretData("evaluations.csv")
	#Genetic
	algo = LearningStrategy(algorithm=Genetics)
	gen_data = algo.learn(data,evaluation,LearningStrategy(algorithm=kNN))	
	#gen_data = algo.learn(data,LearningStrategy(algorithm=kNN))


	#Bayes
	#algo = LearningStrategy(algorithm=BayesNaive)
	#bayes_data = algo.learn(data,evaluation)
	#algo.showResult()

	#kNN
	#algo = LearningStrategy(algorithm=kNN)
	#knn_data = algo.learn(data2)
	#algo.showResult()


if __name__ == "__main__":main()