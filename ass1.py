import math as m
class Mathtools:
    def __init__(self):
        self.total_calls=0
        
    def derivative(self,f,x,h=0.00001):
        self.total_calls+=1
        return ((f(x+h)-f(x))/h)
    
    def gradient(self,f,p,h=0.00001): #here p is the point which can have as many variables as we like
        self.total_calls+=1
        gradient=[]
        for i in p:
            der_i=(f(i+h)-f(i))/h
            gradient.append(der_i)
        return gradient

#now lets define a function as an example

def f(x):
    return (((x**3)/3)+(x**2)+(3*x))

math_tools = Mathtools()

D=math_tools.derivative(f,3)
print("The value of derivative at 3 is :", D)
print('\n')
G=math_tools.gradient(f,[1,2,3,4,5])
print("The value of gradient for points (1,2,3,4,5) are :", G)
print('\n')
print("The total no of times I called the functions :", math_tools.total_calls)