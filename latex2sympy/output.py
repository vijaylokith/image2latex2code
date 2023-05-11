from sympy import *


V = symbols('V')

b = symbols('b')

k = symbols('k')

alpha = symbols('alpha')

c_G = symbols('c_G')

j = symbols('j')

q = symbols('q')

rho = symbols('rho')

equation = Eq(V, alpha*j*k*q/(b*c_G*rho))

dependent = input('Enter the dependent variable: ')

solution = solve(equation, dependent)

variable_value = {}

variables = list(equation.atoms(Symbol))

variables = sorted(list(map(str,variables)))

for variable in variables:
    if variable != dependent:
        variable_value[variable] = float(input(f'Enter the value for {variable}: '))

variable_value = list(variable_value.items())

print(f'The value for {dependent} is: {solution[0].subs(variable_value).evalf()}')