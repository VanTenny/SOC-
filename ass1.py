class Mathtools:
    def __init__(self):
        self.total_calls=0
        
    def derivative(self,f,x,h=0.00001):
        self.total_calls+=1
        return ((f(x+h)-f(x-h))/(2*h))
    
    def gradient(self,f,v,val,h=0.00001): #here v is variable and val is the variable's value
        self.total_calls+=1
        gradients={}
        for var in v:
            index = v.index(var)
            val_plus = val.copy()
            val_minus = val.copy()
            val_plus[index] += h
            val_minus[index] -= h
            grd_i = (f(*val_plus) - f(*val_minus)) / (2 * h)
            gradients[var] = grd_i
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