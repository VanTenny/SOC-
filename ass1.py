class Mathtools:
    def __init__(self):
        self.total_calls=0
        
    def derivative(self,f,x,h=0.00001):
        self.total_calls+=1
        return ((f(x+h)-f(x-h))/(2*h))
    
    def gradient(self,f,variables,values,h=0.00001): 
        self.total_calls+=1
        gradients={}
        for variable in variables: # for each x, y, z
            index = variables.index(variable) # get the index e.g. 0 -> x, 1 -> y, 2 -> z
            val1 = values.copy() 
            val2 = values.copy()
            val1[index] += h # copies the values list and change only one value of the variable by h
            val2[index] -=h  
            grd_index = (f(*val1) - f(*val2)) / (2*h) # This is equivalent to f(1+h, 2, 3) - f(1-h, 2, 3) / 2h
            gradients[variable] = grd_index
        return gradients

#now lets define a function as an example

def f(x):
    return (((x**3)/3)+(x**2)+(3*x))

def g(x,y,z):
    return x**3 + 2*y*x + y**5 + 4*z

math_tools = Mathtools()
variables = ['x', 'y', 'z']
values = [1, 1, 1] 

D=math_tools.derivative(f,3)
print("The value of derivative at 3 is :", D)
print('\n')
G=math_tools.gradient(g,variables,values)
print("The value of gradient for points (1,1,1) are :", G)
print('\n')
print("The total no of times I called the functions :", math_tools.total_calls)