from latex2sympy2 import latex2sympy, latex2latex
import sympy as sp
import string
import re
import pydash as pd

# eq= r"0.3865 + 95.3 \phi^{2} - 12.6 \phi (\frac{r}{r_{o}})^{2} - 9.17 \times 10^{5} \phi (\frac{c}{r_{o}})^3 + 27.8 (\frac{r}{r_{o}})^2 (\frac{c}{r_{o}}) - 2.75 \times 10^3 (\frac{r}{r_{o}})^4 (\frac{c}{r_{o}})^2 + 6.51 \times 10^4 (\frac{r}{r_{o}})^6 (\frac{c}{r_{o}})^2 + 2.13 \phi^{0.5} - 6.59 (\frac{c}{r_{o}}) = k" 
# a,b,c,d,r = sp.symbols("a b c d r")
eq = r"abc=d"
# sy = latex2sympy(eq)
# print(sy)
# p = sy[0]
# print(p)
# x= sp.solve(p,r)
# dic = {'a':2, 'b':2, 'c':2, 'd':2}
# ls = list(dic.items())
# print(x[0].subs(ls))

# x = sp.symbols("x")
# b= 1/(1 + sp.exp(-x))
# d = b.subs(x, 2)
# print(d)

def delta_processing(equation):
    while(pd.strings.count_substr(equation, '\delta') != 0 ):
        start_index = equation.find('\delta')
        rep_string = "\delta " + equation[start_index+7]
        mod_string = "\delta_{" + equation[start_index+7] + "}"
        equation = pd.strings.reg_exp_replace(equation, rep_string, mod_string, count=1)
    
    return equation


def variable_list(equation):
    sympy_equation = latex2sympy(equation)
    if type(sympy_equation) == list:
        return_variables = list(sympy_equation[0].atoms(sp.Symbol))
    else:
        return_variables = list(sympy_equation.atoms(sp.Symbol))
    return_variables = sorted(list(map(str,return_variables)))
    return return_variables



def solve(equation):

    #delta pre-procesing
    # if pd.strings.has_substr(equation, "\delta"):
    #     equation = delta_processing(equation)
    
    # created the output file
    file = open("output.py", "w")
    #importing the sympy library in the output file
    file.write("from sympy import *\n\n\n")
    # comverting the equation into sympy
    sympy_equation = latex2sympy(equation)

    # Seperating the varibles from the expression
    delimiters = "(",")","+","-","*","/"," ",","
    re_pattern = '|'.join(map(re.escape, delimiters))
    variables = set(re.split(re_pattern, str(sympy_equation)))

    #returning list
    return_variables = list(sympy_equation[0].atoms(sp.Symbol))
    return_variables = sorted(list(map(str,return_variables)))

    # Creating a list of all alphabets for variable classification
    # variables_value = {}
    variable_list = string.ascii_letters
    greek_list = ['alpha', 'beta', 'gamma', 'delta', 'varepsilon', 'zeta', 'theta', 'vartheta', 'iota', 'kappa', 'lambda', 'mu ','nu', 'xi', 'pi', 'varpi', 'rho', 'varrho', 'sigma', 'varsigma', 'tau', 'upsilon', 'phi', 'varphi', 'chi','psi', 'omega', 'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon', 'Phi', 'Psi', 'Omega']
    count = 0
    print(str(sympy_equation))
    print(variables)    
    for variable in variables:
        if variable in variable_list and variable != 'e' and variable != '':
            file.write(f"{variable} = symbols('{variable}')\n\n")
            count+=1
        
        elif variable in greek_list:
            file.write(f"{variable} = symbols('{variable}')\n\n")
            count+=1
        
        elif pd.strings.count_substr(variable,'_') == 1 and pd.strings.count_substr(variable,'{') == 0:
            file.write(f"{variable} = symbols('{variable}')\n\n")
            count+=1
    
    if type(sympy_equation) == list:
        file.write(f"equation = {sympy_equation[0]}\n\n")
        eq = sympy_equation[0]
    else:
        file.write(f"equation = {sympy_equation}\n\n")
        eq = sympy_equation

    if pd.strings.count_substr(eq,'Derivative') != 0 or pd.strings.count_substr(eq,'Integral'):
        file.write("result = equation.doit()\n\n")
        file.write("print(f'The solution for the equation is: {result}')")
    elif count > 1:
        file.write("dependent = input('Enter the dependent variable: ')\n\n")
        file.write("solution = solve(equation, dependent)\n\n")
        file.write("variable_value = {}\n\n")
        file.write("variables = list(equation.atoms(Symbol))\n\n")
        file.write("variables = sorted(list(map(str,variables)))\n\n")
        file.write("for variable in variables:\n")
        file.write("    if variable != dependent:\n")
        file.write("        variable_value[variable] = float(input(f'Enter the value for {variable}: '))\n\n")
        file.write("variable_value = list(variable_value.items())\n\n")
        file.write("print(f'The value for {dependent} is: {solution[0].subs(variable_value).evalf()}')")

    else:
        file.write("result = solve(equation, dict=False)\n\n")
        file.write("print(f'The roots for the equation are: {result}')")


    file.close()
    return return_variables



def solve_multi(equation: dict) -> None:
    
    primary_equation = equation["primary"]
    dependent_equations = equation["dependent"]
    num_dep = len(dependent_equations)

    primary_sympy = latex2sympy(primary_equation)[0]
    dependent_sympy = []

    for eq in dependent_equations:
        temp_eq = latex2sympy(eq)
        dependent_sympy.append(temp_eq[0])
    
    pri_eq_variables = list(primary_sympy.atoms(sp.Symbol))
    
    dep_eq_variables = []

    for eq_var in dependent_sympy:
        dep_eq_variables.append(list(eq_var.atoms(sp.Symbol)))

    dep_var_list = {}
    dep_eq_variables_copy = dep_eq_variables.copy()
    dependent_sympy_copy = dependent_sympy.copy()
    for pri_var in pri_eq_variables:
        for i, sec_var_lst in enumerate(dep_eq_variables):
            if pri_var in sec_var_lst:
                sol = sp.solve(dependent_sympy[i], pri_var)
                primary_sympy = primary_sympy.subs(pri_var, sol[0])
                dep_eq_variables_copy.pop(i)
                dependent_sympy_copy.pop(i)
                break

    pri_eq_variables = list(primary_sympy.atoms(sp.Symbol))
    dep_eq_variables = dep_eq_variables_copy.copy()
    dependent_sympy = dependent_sympy_copy.copy()

    for pri_var in pri_eq_variables:
        for i, sec_var_lst in enumerate(dep_eq_variables):
            if pri_var in sec_var_lst:
                sol = sp.solve(dependent_sympy[i], pri_var)
                primary_sympy = primary_sympy.subs(pri_var, sol[0])
                break


    file = open("output.py", "w")
    #importing the sympy library in the output file
    file.write("from sympy import *\n\n\n")

    #returning list
    # return_variables = list(sympy_equation[0].atoms(sp.Symbol))
    # return_variables = sorted(list(map(str,return_variables)))

    # Creating a list of all alphabets for variable classification
    pri_eq_var = list(primary_sympy.atoms(sp.Symbol))

    for var in pri_eq_var:
        file.write(f"{var} = symbols('{var}')\n\n")
    
    count = len(pri_eq_var)

    
    if type(primary_sympy) == list:
        file.write(f"equation = {primary_sympy[0]}\n\n")
    else:
        file.write(f"equation = {primary_sympy}\n\n")

    if count > 1:
        file.write("dependent = input('Enter the dependent variable: ')\n\n")
        file.write("solution = solve(equation, dependent)\n\n")
        file.write("variable_value = {}\n\n")
        file.write("variables = list(equation.atoms(Symbol))\n\n")
        file.write("variables = sorted(list(map(str,variables)))\n\n")
        file.write("for variable in variables:\n")
        file.write("    if variable != dependent:\n")
        file.write("        variable_value[variable] = float(input(f'Enter the value for {variable}: '))\n\n")
        file.write("variable_value = list(variable_value.items())\n\n")
        file.write("print(f'The value for {dependent} is: {solution[0].subs(variable_value).evalf()}')")

    else:
        file.write("result = solve(equation, dict=False)\n\n")
        file.write("print(f'The roots for the equation are: {result}')")


    file.close()

    pri_eq_var = list(map(str, pri_eq_var))
    return pri_eq_var
    

    
# dcs= {"primary": "bcm/f = a", "dependent": ["m/g = d", "r/v = g"]}
# solve_multi(dcs)