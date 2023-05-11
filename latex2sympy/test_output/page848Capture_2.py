from sympy import *


alpha = symbols('alpha')

M_T = symbols('M_T')

b_T = symbols('b_T')

A = symbols('A')

c_T = symbols('c_T')

equation = Eq(c_T, A*alpha/(M_T*b_T))

dependent = input('enter the dependent variable: ')

solution = solve(equation, dependent)

variable_value = {}

for variable in equation.atoms(Symbol):
    if str(variable) != dependent:
        variable_value[variable] = float(input(f'Enter the value for {variable}: '))

variable_value = list(variable_value.items())

print(f'The value for {dependent} is: {solution[0].subs(variable_value).evalf()}')