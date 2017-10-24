from Utils import Util
from Calculator import *
from itertools import combinations
import math
import random
import operator
import numpy as np

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
