from Utils import Util
from Calculator import *
from itertools import combinations
from learningstrategy import LearningStrategy
from knn import kNN
from geneticalgorithm import Genetics
import math
import random
import operator
import numpy as np

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
	data = Util.readInterpretData('evaluation-dataset.csv')
	data2 = Util.readInterpretData('evaluation-dataset.csv')
	evaluation = Util.readInterpretData("evaluation-evaluation.csv")
	#Genetic
	#algo = LearningStrategy(algorithm=Genetics)
	#gen_data = algo.learn(data,evaluation,LearningStrategy(algorithm=BayesNaive))
	#gen_data = algo.learn(data,LearningStrategy(algorithm=kNN))


	#Bayes
	#algo = LearningStrategy(algorithm=BayesNaive)
	#bayes_data = algo.learn(data,evaluation)
	#algo.showResult()

	#kNN
	algo = LearningStrategy(algorithm=kNN)
	knn_data = algo.learn(data2,evaluation)
	algo.showResult()


if __name__ == "__main__":main()
