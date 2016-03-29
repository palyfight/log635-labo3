import csv
from Log635Data import LOG635DATA

class Util:
	def readInterpretData(filename):
		with open(filename) as file:
			reader = csv.DictReader(file,delimiter=';')
			data = []
			for row in reader:
				data.append(row)
		return data

	def bitConverter(row):
		chromosome = ""
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

		for key in sorted(row.keys()):
			func = switcher.get(key)
			chromosome += str(func(row[key]))

		
		return chromosome

	def writeToLearningFile(filename,bits):
		file = open(filename, 'a')
		file.write(bits)
		file.close()
		return ("done writing dataset to file")



