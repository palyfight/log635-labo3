from Utils import Util
from Calculator import *
from itertools import combinations
from individual import Individual
import math
import random
import operator
import numpy as np

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
