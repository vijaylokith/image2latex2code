from sympy import *


H_min = symbols('H_min')

V = symbols('V')

rho = symbols('rho')

M_W = symbols('M_W')

V_min = symbols('V_min')

equation = Eq(rho, -M_W/(H_min*V_min - V + V_min))

dependent = input('enter the dependent variable: ')

solution = solve(equation, dependent)

variable_value = {}

for variable in equation.atoms(Symbol):
    if str(variable) != dependent:
        variable_value[variable] = float(input(f'Enter the value for {variable}: '))

variable_value = list(variable_value.items())

print(f'The value for {dependent} is: {solution[0].subs(variable_value).evalf()}')