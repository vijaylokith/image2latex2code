from sympy import *


p_b = symbols('p_b')

Delta = symbols('Delta')

p_a = symbols('p_a')

p = symbols('p')

equation = Eq(Delta, (-p_a + p_b)/p)

dependent = input('enter the dependent variable: ')

solution = solve(equation, dependent)

variable_value = {}

for variable in equation.atoms(Symbol):
    if str(variable) != dependent:
        variable_value[variable] = float(input(f'Enter the value for {variable}: '))

variable_value = list(variable_value.items())

print(f'The value for {dependent} is: {solution[0].subs(variable_value).evalf()}')