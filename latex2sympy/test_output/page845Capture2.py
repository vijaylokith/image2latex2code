from sympy import *


M_min = symbols('M_min')

H_min = symbols('H_min')

M = symbols('M')

equation = M_min*(H_min/1 + 1) <= M

dependent = input('enter the dependent variable: ')

solution = solve(equation, dependent)

variable_value = {}

for variable in equation.atoms(Symbol):
    if str(variable) != dependent:
        variable_value[variable] = float(input(f'Enter the value for {variable}: '))

variable_value = list(variable_value.items())

print(f'The value for {dependent} is: {solution.subs(variable_value)}')