import csv
from Calculator import *
from Log635Data import LOG635DATA
import numpy as np

class Util:
	def readInterpretData(filename):
		with open(filename) as file:
			reader = csv.DictReader(file,delimiter=';')
			data = []
			for row in reader:
				data.append(row)
		return data

	def splitDataset(dataset, splitRatio):
		trainSize = int(len(dataset) * splitRatio)
		trainSet = []
		np.random.seed(1)
		copy = list(dataset)
		while len(trainSet) < trainSize:
			index = np.random.random_integers(len(copy))
			trainSet.append(copy.pop(index))
		return [trainSet, copy]

	def interpretData(data):
		transformedData = []
		dataDealer = LOG635DATA()

		switcher = {
			'Dalc': dataDealer.Dalc ,
			'Fedu': dataDealer.Fedu ,
			'Fjob': dataDealer.Fjob ,
			'G1': dataDealer.G1 ,
			'G2': dataDealer.G2 ,
			'G3': dataDealer.G3 ,
			'Medu': dataDealer.Medu ,
			'Mjob': dataDealer.Mjob ,
			'Pstatus': dataDealer.Pstatus ,
			'Walc': dataDealer.Walc ,
			'absences': dataDealer.absences ,
			'activities': dataDealer.activities ,
			'address': dataDealer.address ,
			'age': dataDealer.age ,
			'failures': dataDealer.failures ,
			'famrel': dataDealer.famrel ,
			'famsize': dataDealer.famsize ,
			'famsup': dataDealer.famsup ,
			'freetime': dataDealer.freetime ,
			'goout': dataDealer.goout ,
			'guardian': dataDealer.guardian ,
			'health': dataDealer.health ,
			'higher': dataDealer.higher ,
			'internet': dataDealer.internet ,
			'nursery': dataDealer.nursery ,
			'paid': dataDealer.paid ,
			'reason': dataDealer.reason ,
			'romantic': dataDealer.romantic ,
			'school': dataDealer.school ,
			'schoolsup': dataDealer.schoolsup ,
			'sex': dataDealer.sex ,
			'studytime': dataDealer.studytime ,
			'traveltime': dataDealer.traveltime ,
		}	

		for key in sorted(data.keys()):
			func = switcher.get(key)
			if func != None:
				value = func(data[key])
				transformedData.append(int(value,2))


		return transformedData

	def writeToLearningFile(filename,bits):
		file = open(filename, 'a')
		file.write(bits)
		file.close()
		return ("done writing dataset to file")



