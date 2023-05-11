from sympy import *


n = symbols('n')

y = symbols('y')

i = symbols('i')

x_i = symbols('x_i')

equation = Eq(y, x_i*Sum(1, (i, 1, n)))

dependent = input('enter the dependent variable: ')

solution = solve(equation, dependent)

variable_value = {}

for variable in equation.atoms(Symbol):
    if str(variable) != dependent:
        variable_value[variable] = float(input(f'Enter the value for {variable}: '))

variable_value = list(variable_value.items())

print(f'The value for {dependent} is: {solution[0].subs(variable_value).evalf()}')