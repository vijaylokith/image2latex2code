from sympy import *


a_T = symbols('a_T')

c_T = symbols('c_T')

M_T = symbols('M_T')

equation = Eq(M_T, 1/(a_T*c_T))

dependent = input('enter the dependent variable: ')

solution = solve(equation, dependent)

variable_value = {}

for variable in equation.atoms(Symbol):
    if str(variable) != dependent:
        variable_value[variable] = float(input(f'Enter the value for {variable}: '))

variable_value = list(variable_value.items())

print(f'The value for {dependent} is: {solution[0].subs(variable_value).evalf()}')