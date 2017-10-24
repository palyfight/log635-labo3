from Calculator import *
from itertools import combinations
from population import Population
from individual import Individual
import math
import random
import operator
import numpy as np

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
