from sympy import *


rho_TS = symbols('rho_TS')

V_TS = symbols('V_TS')

c_TS = symbols('c_TS')

alpha_TS = symbols('alpha_TS')

A_TS = symbols('A_TS')

b_TS = symbols('b_TS')

equation = Eq(b_TS, A_TS*alpha_TS/(V_TS*c_TS*rho_TS))

dependent = input('enter the dependent variable: ')

solution = solve(equation, dependent)

variable_value = {}

for variable in equation.atoms(Symbol):
    if str(variable) != dependent:
        variable_value[variable] = float(input(f'Enter the value for {variable}: '))

variable_value = list(variable_value.items())

print(f'The value for {dependent} is: {solution[0].subs(variable_value).evalf()}')