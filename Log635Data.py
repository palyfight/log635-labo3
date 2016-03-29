class LOG635DATA:

	def intToBits(self, value, length):
		return bin(int(value))[2:].zfill(length)

	def yesno(self, value):
		if value == 'yes':
			return self.intToBits(1, 1)
		else:
			return self.intToBits(0, 1)

	def Dalc(self, value):
		return self.intToBits(value,3)
		
	def Fedu(self, value):
		return self.intToBits(value,3)

	def Fjob(self, value):
		return self.job(value)

	def G1(self, value):
		return self.intToBits(value,5)

	def G2(self, value):
		return self.intToBits(value,5)

	def G3(self, value):
		return self.intToBits(value,5)

	def Medu(self, value):
		return self.intToBits(value,3)

	def Mjob(self, value):
		return self.job(value)

	def Pstatus(self, value):
		if value == 'T':
			return self.intToBits(1,1)
		else:
			return self.intToBits(0,1)

	def Walc(self, value):
		return self.intToBits(value,3)

	def absences(self, value):
		return self.intToBits(value,7)

	def activities(self, value):
		return self.yesno(value)

	def address(self, value):
		if value == 'U':
			return self.intToBits(0,1)
		elif value == 'R':
			return self.intToBits(1,1)

	def age(self, value):
		return self.intToBits(value,5)

	def failures(self, value):
		return self.intToBits(value,2)

	def famrel(self, value):
		return self.intToBits(value,3)

	def famsize(self, value):
		if value == 'LE3':
			return self.intToBits(0,1)
		elif value == 'GT3':
			return self.intToBits(1,1)

	def famsup(self, value):
		return self.yesno(value)

	def freetime(self, value):
		return self.intToBits(value,3)

	def goout(self, value):
		return self.intToBits(value,3)

	def guardian(self, value):
		if value == 'mother':
			return self.intToBits(1,2)
		elif value == 'father':
			return self.intToBits(2,2)
		elif value == 'other':
			return self.intToBits(3,2)

	def health(self, value):
		return self.intToBits(value,3)

	def higher(self, value):
		return self.yesno(value)

	def internet(self, value):
		return self.yesno(value)

	def nursery(self, value):
		return self.yesno(value)

	def paid(self, value):
		return self.yesno(value)

	def reason(self, value):
		if value == 'home':
			return self.intToBits(1,2)
		elif value == 'reputation':
			return self.intToBits(2,2)
		elif value == 'course':
			return self.intToBits(3,2)
		else:
			return self.intToBits(4,2)

	def romantic(self, value):
		return self.yesno(value)

	def school(self, value):
		if value == 'GP':
			return self.intToBits(1,1)
		else:
			return self.intToBits(0,1)

	def schoolsup(self, value):
		 return self.yesno(value)

	def sex(self, value):
		if value == 'M':
			return self.intToBits(1,1)
		else:
			return self.intToBits(0,1) 

	def studytime(self, value):
		return self.intToBits(value,2)

	def traveltime(self, value):
		return self.intToBits(value,2)

	def job(self, value):
		if value == 'teacher':
			return self.intToBits(1, 3)
		elif value == 'health':
			return self.intToBits(2, 3)
		elif value == 'services':
			return self.intToBits(3, 3)
		elif value == 'at_home':
			return self.intToBits(4, 3)
		else:
			return self.intToBits(5, 3)
	