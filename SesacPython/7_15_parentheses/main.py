import sys 
try:
    from solution.exceptions import InvalidTokenException, NotClosedParenthesesException
except ImportError:
    from exceptions import InvalidTokenException, NotClosedParenthesesException

tokens = ['(', ')']

def find_matching_pair(text, idx):
    """For a given text of parentheses and idx, find the index of matching closing parentheses in the text. 

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
        >> NotClosedParenthesesException """
    s = 0
    flag = False
    
    if not 0 <= idx < len(text):
        raise ValueError(f'Unexpected idx value; (인덱스 범위초과): {idx}')
    if text[idx] == ")": raise NotClosedParenthesesException(f'Cannot close parentheses opened at {idx}; {text}')
    
    for i, char in enumerate(text):
        if char not in tokens:
            raise InvalidTokenException(f'Unexpected token; {char} -> "(",")"만 가능')

        offset = (char == "(")*2 - 1
        s += offset

        if s < 0: raise NotClosedParenthesesException(f'Token at {i} cannot be closed.')
        if i == idx: flag = s
        elif flag and flag-1 == s: return i
    else: raise NotClosedParenthesesException(f'Cannot close parentheses opened at {idx}; {text}')
    """
    # 예시1
    def find_matching_pair(text, idx):
        S = 0
        for i, char in enumerate(text[idx:]):
        if char == '(': S +=1
        else: S -=1
        if S==0:
            return i + idx
    # 예시2
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

def determine_if_rule0(text):
    return text == ''

def determine_if_rule1(text):
    return not determine_if_rule0(text) and find_matching_pair(text, 0) == len(text)-1

def determine_if_rule2(text):
    return not (determine_if_rule0(text) or determine_if_rule1(text))

def parse_empty_string():
    return {'node': '', 'rule': 0}

def default_node_information(text, offset):
    res = {
        'node': text,
        'start': offset,
        'end': len(text) - 1 + offset
        }
    return res
        # 'node': text,
        # 'start': 0,
        # 'end': 1,
        # 'rule': 1,
    
            #     'left': {
            #     'node': '(',
            #     'start': 0,
            #     'end': 0,
            # },
            # 'mid': {
            #     'node': '',
            #     'rule': 0,
            # },
            # 'right': {
            #     'node': ')',
            #     'start': 1,
            #     'end': 1,
            # }

def update_rule1_data(text, res):
    matching_idx = find_matching_pair(text, 0)

    res['rule'] = 1
    res['left'] = {
        'node': '(', 
        'start': 0, 
        'end': 0, 
    }
    res['right'] = {
        'node': ')', 
        'start': matching_idx, 
        'end': matching_idx, 
    }    
    
    return res 

def update_rule1_mid(text, res):
    matching_idx = find_matching_pair(text, 0)
    res['mid'] = parse_parentheses_with_offset(text[1:matching_idx], 1)
    return res 

def update_rule2_data(text, res):
    res['rule'] = 2
    return res 

def update_rule2_nodes(text, res):
    matching_idx = find_matching_pair(text, 0)
    node_indices = [(0, matching_idx)]
    
    while matching_idx != len(text)-1:
        node_start_idx = matching_idx+1 
        if node_start_idx == len(text)-1: break
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

    if rule0:
        return parse_empty_string()
    
    res = default_node_information(text, offset)

    if rule1:
        res = update_rule1_data(text, res)
        res = update_rule1_mid(text, res)
    elif rule2:
        res = update_rule2_data(text, res)
        res = update_rule2_nodes(text, res)
    else:
        assert False, 'Something goes wrong'
    
    return res 

def main():
    if len(sys.argv) != 2:
        print("사용법: python main.py <filename>")
    
    file = sys.argv[1]
    try:
        with open(file, 'r') as f:
            text = f.read().strip()
            print(parse_parentheses(text))
    except FileNotFoundError:print(f'파일을 찾을 수 없습니다: {file}')
    except Exception as e: print(f'오류발생: {e}')

if __name__ == '__main__':
    main()
    from util import print_parsed_result

    print(find_matching_pair('()()(())', 2))
    print(print_parsed_result(parse_parentheses('(())()()(()())')))