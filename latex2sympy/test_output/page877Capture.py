from sympy import *


b = symbols('b')

h = symbols('h')

b_tilde = symbols('b_tilde')

pi = symbols('pi')

phi = symbols('phi')

equation = Eq(phi, pi*(b - b_tilde)/(2*b - h))

dependent = input('enter the dependent variable: ')

solution = solve(equation, dependent)

variable_value = {}

for variable in equation.atoms(Symbol):
    if str(variable) != dependent:
        variable_value[variable] = float(input(f'Enter the value for {variable}: '))

variable_value = list(variable_value.items())

print(f'The value for {dependent} is: {solution[0].subs(variable_value).evalf()}')