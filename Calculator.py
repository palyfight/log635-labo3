import math
class Calculator:

	def __init__(self,formula=None):
		if(formula):
			self.formula = formula()

	def calculate(self,data):
		if(self.formula):
			return self.formula.calculate(data)

class Mean:
	def calculate(self,data):
		return sum(data)/float(len(data))

class STdev:
	def calculate(self,data):
		calc = Calculator(formula=Mean)
		avg = calc.calculate(data)
		denom = 1 if len(data) == 1 else float(len(data)-1)
		variance = sum(math.pow(x-avg,2) for x in data)/denom
		return math.sqrt(variance) 

class Gaussian:
	def calculate(self,data):
		x,mean,stdev = data
		if stdev != 0:
			exp = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
			return (1 / (math.sqrt(2*math.pi) * stdev)) * exp
		else:
			return 0
