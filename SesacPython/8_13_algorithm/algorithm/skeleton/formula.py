
from global_variables import FUNCTION_DICT, OPERATOR_PRIORITY
from copy import deepcopy

import math
import re

from ADT.stack import Stack
from ADT.tree import Tree 

#-------------------------------------              
# Formula Class
#-------------------------------------

class Formula: 
    ''' Formula is essentially a instance of Tree parsed from string equation. It can also be instantiated using tree. 
    '''
    def __init__(self, 
                    eq, 
                    constant_variable = [], 
                    debug = False):
    
        self.eq = eq
        tree = parse(eq)
        self.tree = tree
        
    def __add__(self, other): 
        return Formula('(' + self.eq + ')+(' + other.eq + ')',)
    
    def __sub__(self, other):
        return Formula('(' + self.eq + ')-(' + other.eq + ')',)
    
    def __mul__(self, other):
        return Formula('(' + self.eq + ')*(' + other.eq + ')',)

    def __div__(self, other):
        return Formula('(' + self.eq + ')/(' + other.eq + ')',)
        
    def __str__(self):
        return self.eq

    
def tokenizer(equation):
    left = equation.replace(' ', '')
    tokens = {\
        'op' : ['^', '+', '-', '*', '+', '/'],
        'unary' :  ['-'],
        'para' : ['(', ')'],
        'num' : [r"[1-9][0-9]*\.?[0-9]*|0",],
        'var' : [r"[a-zA-Z]+_?[0-9]*",], 
        'comma' : [',']}
    
    yield ('', '')                
    
def compress_tok(tokenizer):
    yield from tokenizer
            
    
def recursive_descent(tokens):
    
    operator = Stack()
    operand = Stack()
    idx = expr(operator, operand, tokens, 0)
    res = operand.pop()
    
    return res
    
def expr(operator, operand, tokens, idx):
    # expr := part (binary part)*
        
    return idx
    
def part(operator, operand, tokens, idx):
    # part := num | var 
    #      := "(" expr ")"
    #      := func "(" (expr ,)* expr ")"
    #      := unary part 
    
    next_tok = tokens[idx]
    
    return idx
    
def parse(eq):
    return recursive_descent(list(compress_tok(tokenizer(eq))))

    
        
if __name__ == '__main__':
    
    # tests, tests, more tests! 
    
    # simple numbers
    eq1 = '(1)'
    eq2 = '3'
    eq3 = '-1'
    
    # +,- 
    eq4 = '1+1'
    eq5 = '1-1-2' # check
    eq6 = '-1-2-3-4-5'
    
    # +,-,*,/ 
    eq7 = '1+2/3+2'
    eq8 = '3*4+2'
    eq9 = '4/2'
    eq10 = '3+4*2'
    eq11 = '3+4/2'
    eq12 = '3/4/2'
    eq13 = '(3/4)/2' # check
    eq14 = '3/(4/2)'
    eq15 = '1+2/3'
    
    # +,-,*,/,^ with (,)
    eq16 = '(1+2)/3'
    eq17 = '(1*2)/3'
    eq18 = '(1+2)*3'
    eq19 = '3*(1+2)'
    eq20 = '3*(2-1)'
    eq21 = '3*(1-2)'
    eq22 = '3*(-2+1)'
    eq23 = '-3-2^3'
    eq24 = '-3-2^(3+2)'
    eq25 = '-2^3'
    eq26 = '-2^-3'
    
    # +,-,*,/ with nested (,)
    eq27 = '-1+(-1-2)'
    eq28 = '-(2+2)'
    eq29 = '3+(2^(-(2+2)))'
    eq30 = '3*(2*2+1)'
    eq31 = '2-3*(2*2+1)'
    eq32 = '2-3*(2*(2+1))'
    eq33 = '((3+2)*4-(2*4+2^(2-5)))*(2+(3+2)*5^2)'
    eq34 = '2+(3+2)*5^2'
    eq35 = '1+2^2*1'
    
    eq36 = 'x'
    # eq37 = '-x_0*z+y'
    eq40 = '1+3^3*c'
    eq45 = 'a+b+C+d+e+f+g+h'
    eq46 = '1'
    eq47 = '0'
    
    eq48 = 'sin(x+y)'
    for t, tok in compress_tok(tokenizer(eq48)):
        print(t, tok)
       
    print(parse(eq48))
    
    
    for i in range(100):
        try:
            eq = eval('eq%d'%i)
        except NameError:
            continue
        print('=============')
        print(eq)
        for t, tok in compress_tok(tokenizer(eq)):
            print(t, tok)
        print(parse(eq))
        print('=============')
    
"""
func sin
para (
var x
op +
var y
para )
('func', 'sin')
└── ('op', '+')
    ├── ('var', 'x')
    └── ('var', 'y')
=============
(1)
para (
num 1
para )
('num', '1')
=============
=============
3
num 3
('num', '3')
=============
=============
-1
unary unary -
num 1
('num', '-1')
=============
=============
1+1
num 1
op +
num 1
('op', '+')
├── ('num', '1')
└── ('num', '1')
=============
=============
1-1-2
num 1
op +
unary unary -
num 1
op +
unary unary -
num 2
('op', '+')
├── ('num', '1')
└── ('op', '+')
    ├── ('num', '-1')
    └── ('num', '-2')
=============
=============
-1-2-3-4-5
unary unary -
num 1
op +
unary unary -
num 2
op +
unary unary -
num 3
op +
unary unary -
num 4
op +
unary unary -
num 5
('op', '+')
├── ('num', '-1')
└── ('op', '+')
    ├── ('num', '-2')
    └── ('op', '+')
        ├── ('num', '-3')
        └── ('op', '+')
            ├── ('num', '-4')
            └── ('num', '-5')
=============
=============
1+2/3+2
num 1
op +
num 2
op /
num 3
op +
num 2
('op', '+')
├── ('num', '1')
└── ('op', '+')
    ├── ('op', '/')
    │   ├── ('num', '2')
    │   └── ('num', '3')
    └── ('num', '2')
=============
=============
3*4+2
num 3
op *
num 4
op +
num 2
('op', '+')
├── ('op', '*')
│   ├── ('num', '3')
│   └── ('num', '4')
└── ('num', '2')
=============
=============
4/2
num 4
op /
num 2
('op', '/')
├── ('num', '4')
└── ('num', '2')
=============
=============
3+4*2
num 3
op +
num 4
op *
num 2
('op', '+')
├── ('num', '3')
└── ('op', '*')
    ├── ('num', '4')
    └── ('num', '2')
=============
=============
3+4/2
num 3
op +
num 4
op /
num 2
('op', '+')
├── ('num', '3')
└── ('op', '/')
    ├── ('num', '4')
    └── ('num', '2')
=============
=============
3/4/2
num 3
op /
num 4
op /
num 2
('op', '/')
├── ('num', '3')
└── ('op', '/')
    ├── ('num', '4')
    └── ('num', '2')
=============
=============
(3/4)/2
para (
num 3
op /
num 4
para )
op /
num 2
('op', '/')
├── ('op', '/')
│   ├── ('num', '3')
│   └── ('num', '4')
└── ('num', '2')
=============
=============
3/(4/2)
num 3
op /
para (
num 4
op /
num 2
para )
('op', '/')
├── ('num', '3')
└── ('op', '/')
    ├── ('num', '4')
    └── ('num', '2')
=============
=============
1+2/3
num 1
op +
num 2
op /
num 3
('op', '+')
├── ('num', '1')
└── ('op', '/')
    ├── ('num', '2')
    └── ('num', '3')
=============
=============
(1+2)/3
para (
num 1
op +
num 2
para )
op /
num 3
('op', '/')
├── ('op', '+')
│   ├── ('num', '1')
│   └── ('num', '2')
└── ('num', '3')
=============
=============
(1*2)/3
para (
num 1
op *
num 2
para )
op /
num 3
('op', '/')
├── ('op', '*')
│   ├── ('num', '1')
│   └── ('num', '2')
└── ('num', '3')
=============
=============
(1+2)*3
para (
num 1
op +
num 2
para )
op *
num 3
('op', '*')
├── ('op', '+')
│   ├── ('num', '1')
│   └── ('num', '2')
└── ('num', '3')
=============
=============
3*(1+2)
num 3
op *
para (
num 1
op +
num 2
para )
('op', '*')
├── ('num', '3')
└── ('op', '+')
    ├── ('num', '1')
    └── ('num', '2')
=============
=============
3*(2-1)
num 3
op *
para (
num 2
op +
unary unary -
num 1
para )
('op', '*')
├── ('num', '3')
└── ('op', '+')
    ├── ('num', '2')
    └── ('num', '-1')
=============
=============
3*(1-2)
num 3
op *
para (
num 1
op +
unary unary -
num 2
para )
('op', '*')
├── ('num', '3')
└── ('op', '+')
    ├── ('num', '1')
    └── ('num', '-2')
=============
=============
3*(-2+1)
num 3
op *
para (
unary unary -
num 2
op +
num 1
para )
('op', '*')
├── ('num', '3')
└── ('op', '+')
    ├── ('num', '-2')
    └── ('num', '1')
=============
=============
-3-2^3
unary unary -
num 3
op +
unary unary -
num 2
op ^
num 3
('op', '+')
├── ('num', '-3')
└── ('op', '^')
    ├── ('num', '-2')
    └── ('num', '3')
=============
=============
-3-2^(3+2)
unary unary -
num 3
op +
unary unary -
num 2
op ^
para (
num 3
op +
num 2
para )
('op', '+')
├── ('num', '-3')
└── ('op', '^')
    ├── ('num', '-2')
    └── ('op', '+')
        ├── ('num', '3')
        └── ('num', '2')
=============
=============
-2^3
unary unary -
num 2
op ^
num 3
('op', '^')
├── ('num', '-2')
└── ('num', '3')
=============
=============
-2^-3
unary unary -
num 2
op ^
unary unary -
num 3
('op', '^')
├── ('num', '-2')
└── ('num', '-3')
=============
=============
-1+(-1-2)
unary unary -
num 1
op +
para (
unary unary -
num 1
op +
unary unary -
num 2
para )
('op', '+')
├── ('num', '-1')
└── ('op', '+')
    ├── ('num', '-1')
    └── ('num', '-2')
=============
=============
-(2+2)
unary unary -
para (
num 2
op +
num 2
para )
('op', '*')
├── ('num', -1)
└── ('op', '+')
    ├── ('num', '2')
    └── ('num', '2')
=============
=============
3+(2^(-(2+2)))
num 3
op +
para (
num 2
op ^
para (
unary unary -
para (
num 2
op +
num 2
para )
para )
para )
('op', '+')
├── ('num', '3')
└── ('op', '^')
    ├── ('num', '2')
    └── ('op', '*')
        ├── ('num', -1)
        └── ('op', '+')
            ├── ('num', '2')
            └── ('num', '2')
=============
=============
3*(2*2+1)
num 3
op *
para (
num 2
op *
num 2
op +
num 1
para )
('op', '*')
├── ('num', '3')
└── ('op', '+')
    ├── ('op', '*')
    │   ├── ('num', '2')
    │   └── ('num', '2')
    └── ('num', '1')
=============
=============
2-3*(2*2+1)
num 2
op +
unary unary -
num 3
op *
para (
num 2
op *
num 2
op +
num 1
para )
('op', '+')
├── ('num', '2')
└── ('op', '*')
    ├── ('num', '-3')
    └── ('op', '+')
        ├── ('op', '*')
        │   ├── ('num', '2')
        │   └── ('num', '2')
        └── ('num', '1')
=============
=============
2-3*(2*(2+1))
num 2
op +
unary unary -
num 3
op *
para (
num 2
op *
para (
num 2
op +
num 1
para )
para )
('op', '+')
├── ('num', '2')
└── ('op', '*')
    ├── ('num', '-3')
    └── ('op', '*')
        ├── ('num', '2')
        └── ('op', '+')
            ├── ('num', '2')
            └── ('num', '1')
=============
=============
((3+2)*4-(2*4+2^(2-5)))*(2+(3+2)*5^2)
para (
para (
num 3
op +
num 2
para )
op *
num 4
op +
unary unary -
para (
num 2
op *
num 4
op +
num 2
op ^
para (
num 2
op +
unary unary -
num 5
para )
para )
para )
op *
para (
num 2
op +
para (
num 3
op +
num 2
para )
op *
num 5
op ^
num 2
para )
('op', '*')
├── ('op', '+')
│   ├── ('op', '*')
│   │   ├── ('op', '+')
│   │   │   ├── ('num', '3')
│   │   │   └── ('num', '2')
│   │   └── ('num', '4')
│   └── ('op', '*')
│       ├── ('num', -1)
│       └── ('op', '+')
│           ├── ('op', '*')
│           │   ├── ('num', '2')
│           │   └── ('num', '4')
│           └── ('op', '^')
│               ├── ('num', '2')
│               └── ('op', '+')
│                   ├── ('num', '2')
│                   └── ('num', '-5')
└── ('op', '+')
    ├── ('num', '2')
    └── ('op', '*')
        ├── ('op', '+')
        │   ├── ('num', '3')
        │   └── ('num', '2')
        └── ('op', '^')
            ├── ('num', '5')
            └── ('num', '2')
=============
=============
2+(3+2)*5^2
num 2
op +
para (
num 3
op +
num 2
para )
op *
num 5
op ^
num 2
('op', '+')
├── ('num', '2')
└── ('op', '*')
    ├── ('op', '+')
    │   ├── ('num', '3')
    │   └── ('num', '2')
    └── ('op', '^')
        ├── ('num', '5')
        └── ('num', '2')
=============
=============
1+2^2*1
num 1
op +
num 2
op ^
num 2
op *
num 1
('op', '+')
├── ('num', '1')
└── ('op', '*')
    ├── ('op', '^')
    │   ├── ('num', '2')
    │   └── ('num', '2')
    └── ('num', '1')
=============
=============
x
var x
('var', 'x')
=============
=============
1+3^3*c
num 1
op +
num 3
op ^
num 3
op *
var c
('op', '+')
├── ('num', '1')
└── ('op', '*')
    ├── ('op', '^')
    │   ├── ('num', '3')
    │   └── ('num', '3')
    └── ('var', 'c')
=============
=============
a+b+C+d+e+f+g+h
var a
op +
var b
op +
var C
op +
var d
op +
var e
op +
var f
op +
var g
op +
var h
('op', '+')
├── ('var', 'a')
└── ('op', '+')
    ├── ('var', 'b')
    └── ('op', '+')
        ├── ('var', 'C')
        └── ('op', '+')
            ├── ('var', 'd')
            └── ('op', '+')
                ├── ('var', 'e')
                └── ('op', '+')
                    ├── ('var', 'f')
                    └── ('op', '+')
                        ├── ('var', 'g')
                        └── ('var', 'h')
=============
=============
1
num 1
('num', '1')
=============
=============
0
num 0
('num', '0')
=============
=============
sin(x+y)
func sin
para (
var x
op +
var y
para )
('func', 'sin')
└── ('op', '+')
    ├── ('var', 'x')
    └── ('var', 'y')
=============
"""
    
    