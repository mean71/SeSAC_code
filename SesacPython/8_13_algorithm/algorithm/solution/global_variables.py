import math 


OPERATOR_PRIORITY = {\
    '+' : 1, 
    '-' : 1, 
    '*' : 2, 
    '/' : 2, 
    '^' : 3, 
    'unary -' : 4}

PARA = ['(', ')', '[', ']', '{', '}',]


    
FUNCTION_DICT = {\
    'cos' : (math.cos, math.sin, 
                      '-1*sin(placeholder)'),
    'sin' : (math.sin, lambda x:math.cos(x), 
                       'cos(placeholder)',),
    'tan' : (math.tan, lambda x:1/(math.cos(x)**2), 
                       '1/cos(placeholder)^2'),
    'ln' : (math.log, lambda x:1/x, 
                      '1/placeholder'),
    # custom functions 
    }