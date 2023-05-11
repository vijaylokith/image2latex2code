from sympy import *


rho = symbols('rho')

c_G = symbols('c_G')

V = symbols('V')

b = symbols('b')

alpha = symbols('alpha')

A = symbols('A')

equation = Eq(rho, A*alpha/(V*b*c_G))

dependent = input('enter the dependent variable: ')

solution = solve(equation, dependent)

variable_value = {}

for variable in equation.atoms(Symbol):
    if str(variable) != dependent:
        variable_value[variable] = float(input(f'Enter the value for {variable}: '))

variable_value = list(variable_value.items())

print(f'The value for {dependent} is: {solution[0].subs(variable_value).evalf()}')