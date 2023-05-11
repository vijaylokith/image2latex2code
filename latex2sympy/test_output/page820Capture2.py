from sympy import *


V = symbols('V')

M_W = symbols('M_W')

V_min = symbols('V_min')

rho = symbols('rho')

equation = rho*(V - V_min) > M_W

dependent = input('enter the dependent variable: ')

solution = solve(equation, dependent)

variable_value = {}

for variable in equation.atoms(Symbol):
    if str(variable) != dependent:
        variable_value[variable] = float(input(f'Enter the value for {variable}: '))

variable_value = list(variable_value.items())

print(f'The value for {dependent} is: {solution.subs(variable_value)}')