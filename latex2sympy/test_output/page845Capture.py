from sympy import *


K = symbols('K')

b = symbols('b')

P = symbols('P')

V = symbols('V')

T = symbols('T')

r = symbols('r')

R_s = symbols('R_s')

p = symbols('p')

M = symbols('M')

equation = Eq(K, -T/27315 + 20000*P*V*p/(5463*M*R_s*b*r))

dependent = input('enter the dependent variable: ')

solution = solve(equation, dependent)

variable_value = {}

for variable in equation.atoms(Symbol):
    if str(variable) != dependent:
        variable_value[variable] = float(input(f'Enter the value for {variable}: '))

variable_value = list(variable_value.items())

print(f'The value for {dependent} is: {solution[0].subs(variable_value).evalf()}')