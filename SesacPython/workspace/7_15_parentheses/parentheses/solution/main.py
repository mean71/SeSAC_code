import sys 
try:
    from solution.exceptions import InvalidTokenException, NotClosedParenthesesException
except ImportError:
    from exceptions import InvalidTokenException, NotClosedParenthesesException

tokens = ['(', ')']

def find_matching_pair(text, idx):
    """For a given text of parentheses and idx, find the index of matching parentheses in the text. 

    Args:
        str text 
        int idx 
    Returns:
        int
    Raises:
        InvalidTokenException: When the input contains invalid character.
        NotClosedParenthesesException: When it is impossible to find the matching parentheses. 
        ValueError: When the input idx is larger or equal to len(text) or is smaller than 0. 
    
    Examples:
        find_matching_pair('()', 0)
        >> 1 
        find_matching_pair('(())', 1)
        >> 2
        find_matching_pair(')', 0)
        >> NotClosedParenthesesException 
### 신승우강사님 예시 답안코드
def find_matching_pair(text, idx):
    S = 0
    for i, char in enumerate(text[idx:]):
    if char == '(': S +=1
    else: S -=1
    if S==0:
        return i + idx
### 두번째
def find_matching_pair(text, idx):
    num = 0
    print(text)
    for i,v in enumerate(text):
        
        if i >= idx:
            if v == '(':
                num += 1
                print('추가')
            if v == ')':
                num -= 1
            if num == 0:
                print(i)
                return i
    """
    
    if idx < 0 or len(text) <= idx:
        raise ValueError(f'Unexpected idx value; {idx}')
    
    s = 0 
    flag = False
    
    for i, char in enumerate(text):
        if char not in tokens:
            raise InvalidTokenException(f'Undexpected token; {char}')
        
        offset = 1 if char == '(' else -1 
        s += offset
        # 위 2줄과 아래 1줄은 같다
        s += 1 if char == '(' else -1 

        if s < 0:
            raise NotClosedParenthesesException(f'Token at {i} cannot be closed.')
        
        if i == idx:
            if char == ')':
                raise NotClosedParenthesesException(f'Cannot close parentheses opened at {idx}; {text}')        
            flag = s 
        elif flag-1 == s and flag:
            return i 
    else:
        raise NotClosedParenthesesException(f'Cannot close parentheses opened at {idx}; {text}')

def determine_if_rule0(text):
    return text == ''

def determine_if_rule1(text):
    return not determine_if_rule0(text) and find_matching_pair(text, 0) == len(text)-1

def determine_if_rule2(text):
    return not (determine_if_rule0(text) or determine_if_rule1(text))

def parse_empty_string():
    return {'node': '', 'rule': 0}

def default_node_information(text, offset):
    res = {}
    
    res['node'] = text 
    res['start'] = offset
    res['end'] = len(text)-1+offset
    
    return res 

def update_rule1_data(text, res):
    assert determine_if_rule1(text)

    matching_idx = find_matching_pair(text, 0)

    res['rule'] = 1        
    res['left'] = {\
        'node': '(', 
        'start': 0, 
        'end': 0, 
    }
    res['right'] = {\
        'node': ')', 
        'start': matching_idx, 
        'end': matching_idx, 
    }    
    
    return res 

def update_rule1_mid(text, res):
    assert determine_if_rule1(text)
    
    matching_idx = find_matching_pair(text, 0)

    res['mid'] = parse_parentheses_with_offset(text[1:matching_idx], 1)
    
    return res 

def update_rule2_data(text, res):
    assert determine_if_rule2(text)

    res['rule'] = 2
    
    return res 

def update_rule2_nodes(text, res):
    assert determine_if_rule2(text)

    matching_idx = find_matching_pair(text, 0)

    node_indices = [(0, matching_idx)]
        
    while matching_idx != len(text)-1:
        node_start_idx = matching_idx+1 
        if node_start_idx == len(text)-1:
            break
        matching_idx = find_matching_pair(text, node_start_idx)
        node_indices.append((node_start_idx, matching_idx))
    
    res['nodes'] = [parse_parentheses_with_offset(text[start:end+1], start) for start, end in node_indices]

    return res 

def parse_parentheses(text):
    """For the given string, parse it in the form of dict. 

    For detailed explanation about the parsing process and the result format, consult parentheses/documents/assignment.txt file. 

    Args:
        str text
    Returns:
        dict 
    Raises:
        InvalidTokenException: When the input contains invalid character.
        NotClosedParenthesesException: When the input have a syntax error.
    Examples:

    parse_parentheses('')
    >> {
            'node': '',
            'rule': 0,  
    }
    parse_parentheses('()')
    >> {
            'node': '()', 
            'start': 0, 
            'end': 1,
            'rule': 1, 
            'left': {
                'node': '(', 
                'start': 0, 
                'end': 0, 
            },
            'mid': {
                'node': '', 
                'rule': 0, 
            }, 
            'right': {
                'node': ')',
                'start': 1, 
                'end': 1,   
            },
    }
    parse_parentheses('(())')
    >> {
            'node': '(())', 
            'start': 0, 
            'end': 3, 
            'rule': 1, 
            'left': {
                'node': '(', 
                'start': 0, 
                'end': 0, 
            }, 
            'mid': {}, // Same as parse_parentheses('()'), except for start/end attributes. 
            'right': {
                'node': ')', 
                'start': 3, 
                'end': 3, 
            }
    }
    parse_parentheses('()()')
    >> {
            'node': '()()', 
            'start': 0, 
            'end': 3, 
            'rule': 2, 
            'nodes': [
                {...},  // Same as parse_parentheses('()').
                {...},  // Same as parse_parentheses('()'), except for start/end attributes. 
            ]
    }
    parse_parentheses('(()())')
    >> {
            'node': '(()())', 
            'start': 0, 
            'end': 5, 
            'rule': 1, 
            'left': {...}, // Same as parse_parentheses('()')['left'] 
            'mid': {...}, // Same as parse_parentheses('()()'), except for start/end attributes. 
            'right': {...}, // Same as parse_parentheses('()')['left'], except for start/end attributes. 
    }
    """ 

    return parse_parentheses_with_offset(text)

def parse_parentheses_with_offset(text, offset = 0):
    # if text == '': # rule 0 
    #     return {'node': '', 'rule': 0}
    
    # res = {}
    # res['node'] = text 
    # res['start'] = offset
    # res['end'] = len(text)-1+offset

    # matching_idx = find_matching_pair(text, 0)

    # if matching_idx == len(text)-1: # rule 1
    #     res['rule'] = 1        
    #     res['left'] = {\
    #         'node': '(', 
    #         'start': 0, 
    #         'end': 0, 
    #     }
    #     res['right'] = {\
    #         'node': ')', 
    #         'start': matching_idx, 
    #         'end': matching_idx, 
    #     }
    #     res['mid'] = parse_parentheses_with_offset(text[1:matching_idx], 1)
    # else: # rule 2 
    #     res['rule'] = 2
    #     node_indices = [(0, matching_idx)]
        
    #     while matching_idx != len(text)-1:
    #         node_start_idx = matching_idx+1 
    #         if node_start_idx == len(text)-1:
    #             break
    #         matching_idx = find_matching_pair(text, node_start_idx)
    #         node_indices.append((node_start_idx, matching_idx))
        
    #     res['nodes'] = [parse_parentheses_with_offset(text[start:end+1], start) for start, end in node_indices]

    # return res 

    rule0 = determine_if_rule0(text)
    rule1 = determine_if_rule1(text) 
    rule2 = determine_if_rule2(text) 

    if rule0: # rule 0 
        return parse_empty_string()
    
    res = default_node_information(text, offset)

    if rule1: # rule 1
        res = update_rule1_data(text, res)
        res = update_rule1_mid(text, res)
    elif rule2: # rule 2 
        res = update_rule2_data(text, res) 
        res = update_rule2_nodes(text, res)     
    else:
        assert False, 'Something goes wrong' 
    
    return res 

def main():
    args = sys.argv
    with open(f'{sys.argv[1]}', 'r') as f:
        text = f.read().strip()
        print(parse_parentheses(text))

if __name__ == '__main__':
    # main()
    from util import print_parsed_result

    print(print_parsed_result(parse_parentheses('(())()()(()())')))