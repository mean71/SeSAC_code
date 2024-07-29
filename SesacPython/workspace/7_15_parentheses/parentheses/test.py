import os 
import sys 
from random import randint, random

testcase_dir = 'testcases'
test_result_file = 'test_result.txt'

def generate_testcases(n, min_length = 10, max_length = 100):
    res = []
    
    for l in range(min_length, max_length + 1):
        for _ in range(n):
            res.append(generate_text(l))

    if not os.path.exists(testcase_dir):
        os.makedirs(testcase_dir)
    
    for idx, elem in enumerate(res):
        with open(f'./{testcase_dir}/testcase{idx}.para', 'w+') as f:
            f.write(elem)

    return res 

def generate_error_testcases(n, min_length = 10, max_length = 100):
    res = []
    
    for l in range(min_length, max_length + 1):
        for _ in range(n):
            res.append(generate_error(l))

    if not os.path.exists(testcase_dir):
        os.makedirs(testcase_dir)
    
    for idx, elem in enumerate(res):
        with open(f'./{testcase_dir}/error_testcase{idx}.para', 'w+') as f:
            f.write(elem)

    return res 

def generate_text(n):
    """Generate a random string with length 2*n of balanced parentheses. 
    """
    if n == 1:
        return '()'
    else:
        if random() > 0.5:
            return '(' + generate_text(n-1) + ')'
        else:
            k = randint(1, n-1)
            return generate_text(k) + generate_text(n-k)

def generate_error(n):
    tmp = ''
    for _ in range(2*n):
        if random() < 0.5:
            tmp += '('
        else:
            tmp += ')'
    
    return tmp 

def testcases(include_error_cases = False):
    for elem in os.listdir(testcase_dir):
        if elem.endswith('.para'):
            if not include_error_cases:
                if elem.startswith('error'):
                   continue  
            with open(os.path.join(testcase_dir, elem), 'r') as f:
                text = f.read().strip()
                yield text 

def check_solution():
    from solution.main import find_matching_pair, parse_parenthesis

    for text in testcases():
        find_matching_pair(text, 0)
        parse_parenthesis(text)

def check_skeleton():
    func_list = ['find_matching_pair', 'determine_if_rule0', 'determine_if_rule1', 'determine_if_rule2', 'parse_empty_string', 'default_node_information', 'update_rule1_data', 'update_rule1_mid', 'update_rule2_data', 'update_rule2_nodes']
    
    for idx, func in enumerate(func_list):
        exec(f'from solution.main import {func} as a{str(idx)}') 
        exec(f'from skeleton.main import {func} as q{str(idx)}') 
    # from solution.main import find_matching_pair as a1 
    # from solution.main import determine_if_rule0 as a2
    # from skeleton.main import find_matching_pair as q1 
    
    test_result = open(test_result_file, 'w+') 
    cases = list(testcases())[:100]
    total = len(cases)
    
    
    i = 0
    # find_matching_pair 
    print(f'=====================================', file = test_result)
    print(f'Test Result of function {func_list[i]}', file = test_result)
    correct = 0
    error = 0
    
    for text in cases:
        t = len(text)
        r = randint(0, t-1)
        while text[r] != '(':
            r = randint(0, t-1)
        a = locals()[f'a{i}'](text, r) 
        q = locals()[f'q{i}'](text, r)
        
        if a == q:
            correct += 1 
        else:
            error += 1 
            print(f'Failed on {text}; expected {str(a)}, got {str(q)}', file = test_result)
    
    print(f'{correct} / {total}', file = test_result)
    print(f'=====================================', file = test_result)
    print(f'=====================================')
    print(f'Test Result of function {func_list[i]}')
    print(f'{correct} / {total}')
    print(f'=====================================')
    i += 1 
    
    # determine_if_rule0
    print(f'=====================================', file = test_result)
    print(f'Test Result of function {func_list[i]}', file = test_result)
    correct = 0
    error = 0
    
    for text in cases:
        a = locals()[f'a{i}'](text) 
        q = locals()[f'q{i}'](text)
        if a == q:
            correct += 1 
        else:
            error += 1 
            print(f'Failed on {text}; expected {str(a)}, got {str(q)}', file = test_result)
    
    print(f'{correct} / {total}', file = test_result)
    print(f'=====================================', file = test_result)
    print(f'=====================================')
    print(f'Test Result of function {func_list[i]}')
    print(f'{correct} / {total}')
    print(f'=====================================')
    i += 1 

    # determine_if_rule1
    print(f'=====================================', file = test_result)
    print(f'Test Result of function {func_list[i]}', file = test_result)
    correct = 0
    error = 0
    
    for text in cases:
        a = locals()[f'a{i}'](text) 
        q = locals()[f'q{i}'](text)
        if a == q:
            correct += 1 
        else:
            error += 1 
            print(f'Failed on {text}; expected {str(a)}, got {str(q)}', file = test_result)
    
    print(f'{correct} / {total}', file = test_result)
    print(f'=====================================', file = test_result)
    print(f'=====================================')
    print(f'Test Result of function {func_list[i]}')
    print(f'{correct} / {total}')
    print(f'=====================================')
    i += 1 

    # determine_if_rule2
    print(f'=====================================', file = test_result)
    print(f'Test Result of function {func_list[i]}', file = test_result)
    correct = 0
    error = 0
    
    for text in cases:
        a = locals()[f'a{i}'](text) 
        q = locals()[f'q{i}'](text)
        if a == q:
            correct += 1 
        else:
            error += 1 
            print(f'Failed on {text}; expected {str(a)}, got {str(q)}', file = test_result)
    
    print(f'{correct} / {total}', file = test_result)
    print(f'=====================================', file = test_result)
    print(f'=====================================')
    print(f'Test Result of function {func_list[i]}')
    print(f'{correct} / {total}')
    print(f'=====================================')
    i += 1 

    # parse_empty_string
    print(f'=====================================', file = test_result)
    print(f'Test Result of function {func_list[i]}', file = test_result)
    correct = 0
    error = 0
    
    for text in cases:
        a = locals()[f'a{i}']() 
        q = locals()[f'q{i}']()
        if a == q:
            correct += 1 
        else:
            error += 1 
            print(f'Failed on {text}; expected {str(a)}, got {str(q)}', file = test_result)
    
    print(f'{correct} / {total}', file = test_result)
    print(f'=====================================', file = test_result)
    print(f'=====================================')
    print(f'Test Result of function {func_list[i]}')
    print(f'{correct} / {total}')
    print(f'=====================================')
    i += 1 

    # default_node_information
    print(f'=====================================', file = test_result)
    print(f'Test Result of function {func_list[i]}', file = test_result)
    correct = 0
    error = 0
    
    for text in cases:
        r = randint(0, 3)    
        a = locals()[f'a{i}'](text, r) 
        q = locals()[f'q{i}'](text, r)
        if a == q:
            correct += 1 
        else:
            error += 1 
            print(f'Failed on {text}; expected {str(a)}, got {str(q)}', file = test_result)
    
    print(f'{correct} / {total}', file = test_result)
    print(f'=====================================', file = test_result)
    print(f'=====================================')
    print(f'Test Result of function {func_list[i]}')
    print(f'{correct} / {total}')
    print(f'=====================================')
    i += 1 

    # update_rule1_data
    print(f'=====================================', file = test_result)
    print(f'Test Result of function {func_list[i]}', file = test_result)
    correct = 0
    error = 0
    
    
    for text in cases:
        try:
            a = locals()[f'a{i}'](text, {}) 
        except Exception as ea:
            a = ea 
        try:
            q = locals()[f'q{i}'](text, {})
        except Exception as eq:
            q = eq 
        if a == q:
            correct += 1 
        else:
            error += 1 
            print(f'Failed on {text}; expected {str(a)}, got {str(q)}', file = test_result)
    
    print(f'{correct} / {total}', file = test_result)
    print(f'=====================================', file = test_result)
    print(f'=====================================')
    print(f'Test Result of function {func_list[i]}')
    print(f'{correct} / {total}')
    print(f'=====================================')
    i += 1 

    # update_rule1_mid 
    print(f'=====================================', file = test_result)
    print(f'Test Result of function {func_list[i]}', file = test_result)
    correct = 0
    error = 0
    
    
    for text in cases:
        try:
            a = locals()[f'a{i}'](text, {}) 
        except Exception as ea:
            a = ea 
        try:
            q = locals()[f'q{i}'](text, {})
        except Exception as eq:
            q = eq 
        if a == q:
            correct += 1 
        else:
            error += 1 
            print(f'Failed on {text}; expected {str(a)}, got {str(q)}', file = test_result)
    
    print(f'{correct} / {total}', file = test_result)
    print(f'=====================================', file = test_result)
    print(f'=====================================')
    print(f'Test Result of function {func_list[i]}')
    print(f'{correct} / {total}')
    print(f'=====================================')
    i += 1 

    # update_rule2_data
    print(f'=====================================', file = test_result)
    print(f'Test Result of function {func_list[i]}', file = test_result)
    correct = 0
    error = 0
    
    
    for text in cases:
        try:
            a = locals()[f'a{i}'](text, {}) 
        except Exception as ea:
            a = ea 
        try:
            q = locals()[f'q{i}'](text, {})
        except Exception as eq:
            q = eq 
        if a == q:
            correct += 1 
        else:
            error += 1 
            print(f'Failed on {text}; expected {str(a)}, got {str(q)}', file = test_result)
    
    print(f'{correct} / {total}', file = test_result)
    print(f'=====================================', file = test_result)
    print(f'=====================================')
    print(f'Test Result of function {func_list[i]}')
    print(f'{correct} / {total}')
    print(f'=====================================')
    i += 1 

    # update_rule2_nodes 
    print(f'=====================================', file = test_result)
    print(f'Test Result of function {func_list[i]}', file = test_result)
    
    correct = 0
    error = 0
    
    for text in cases:
        try:
            a = locals()[f'a{i}'](text, {}) 
        except Exception as ea:
            a = ea 
        try:
            q = locals()[f'q{i}'](text, {})
        except Exception as eq:
            q = eq 
        if a == q:
            correct += 1 
        else:
            error += 1 
            print(f'Failed on {text}; expected {str(a)}, got {str(q)}', file = test_result)
    
    print(f'{correct} / {total}', file = test_result)
    print(f'=====================================', file = test_result)
    print(f'=====================================')
    print(f'Test Result of function {func_list[i]}')
    print(f'{correct} / {total}')
    print(f'=====================================')
    i += 1 

if __name__ == '__main__':
    # generate_testcases(100)
    # generate_error_testcases(10)
    # check_solution()
    check_skeleton()