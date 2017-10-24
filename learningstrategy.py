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
