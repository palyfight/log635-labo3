import csv



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

	class Population:
		pass

	class Individual:
		pass

class Util:
	def readInterpretData(filename):
		with open(filename) as file:
			reader = csv.DictReader(file,delimiter=';')
			data = []
			for row in reader:
				data.append(row)
		return data

def main():
	data = Util.readInterpretData('learning_dataset.csv')
	algo = LearningStrategy(algorithm=Genetics)
	proc_data = algo.learn(data)
	print(proc_data)
if __name__ == "__main__":main()