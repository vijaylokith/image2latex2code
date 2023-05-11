from sympy import *


T_B = symbols('T_B')

P_B = symbols('P_B')

P_A = symbols('P_A')

K = symbols('K')

T_A = symbols('T_A')

equation = Eq(P_A, P_B/(T_B/T_A)**(K/(K - 1)))

dependent = input('enter the dependent variable: ')

solution = solve(equation, dependent)

variable_value = {}

for variable in equation.atoms(Symbol):
    if str(variable) != dependent:
        variable_value[variable] = float(input(f'Enter the value for {variable}: '))

variable_value = list(variable_value.items())

print(f'The value for {dependent} is: {solution[0].subs(variable_value).evalf()}')