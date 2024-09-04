import sys # 파이썬 표준 라이브러리로, 파이썬 인터프리터와 관련된 기능을 제공하는 모듈을 가져옵니다.
# try-except 블록을 사용하여 특정 모듈과 예외 클래스를 가져옵니다.
try:
#'solution.exceptions'모듈from'InvalidTokenException','NotClosedParenthesesException'클래스임포트
    from solution.exceptions import InvalidTokenException, NotClosedParenthesesException
except ImportError:
#만약 위의 임포트가 실패하면, 'exceptions' 모듈에서 다시 동일한 클래스들을 임포트합니다.
#환경에 따라 모듈이 사용불가능할 수 있어 다양한 환경에서도 코드가 작동하도록 하기위함.
    from exceptions import InvalidTokenException, NotClosedParenthesesException
    # InvalidTokenException: 유효하지 않은 토큰이 발견될 때 발생하는 예외클래스
    # NotClosedParenthesesException: 닫히지 않은 괄호가 있을 때 발생하는 예외클래스
#solution모듈:
#토큰리스트는(사용할 특정문자열 집합)를 정의합니다. 이 리스트는 '('와 ')' 두 개의 문자열을 포함합니다.
#이러한 토큰들은 보통 구문분석(parsing)과정에 사용. 예로 특정패턴인식이나, 수식의 괄호를 처리하거나..
tokens = ['(',')']

def find_matching_pair(text, idx):#text와 idx인덱스 매개변수로 idx서 시작하는 괄호의 짝 인덱스반환
    """For a given text of parentheses and idx, find the index of matching parentheses in the text. 
    Args: #독립변수,인수들
        str text 
        int idx  #독립변수?의 자료형
    Returns:
        int # 함수의 리턴값은 정수형
    Raises: # raises '키워드'로 3가지 예외 발생 : 실행용법 - raise (예?외)
        InvalidTokenException: When the input contains invalid character. # 토큰문자열이 아닌듯한
        NotClosedParenthesesException: When it is impossible to find the matching parentheses. # 닫힌괄호매칭이 안될듯한
        ValueError: When the input idx is larger or equal to len(text) or is smaller than 0. # 받은 인덱스가 범위밖인듯한 
    Examples: # 예시
        find_matching_pair('()', 0)
        >> 1 
        find_matching_pair('(())', 1)
        >> 2
        find_matching_pair(')', 0)
        >> NotClosedParenthesesException 
    """
    if 0 <= idx <= len(text):
        for tx in text:            
            if not(tx in tokens):raise InvalidTokenException(f'토큰리스트"(",")"만 가능')
    else:raise ValueError(f'인덱스가 범위바깥')
    offset = 0
    for i,t in enumerate(text):
        if i == idx and t == ')':raise NotClosedParenthesesException(f'맞는짝이 없음')
        offset += +1 if i >= idx and t == '(' else -1
        if offset == 0:                
            return i

#룰적용여부 판단함수는 상단목적과 하단 룰을 참조?
#출력결과값res 함수는 하단 결과값을 참조?
def determine_if_rule0(text): # 룰0를 적용할 조건인지 결정하는 함수
    return text == ''
def determine_if_rule1(text): # 룰1를 적용할 조건인지 결정하는 함수
    return find_matching_pair(text, 0) == len(text)-1
def determine_if_rule2(text): # 룰2를 적용할 조건인지 체크하는 함수
    return not(determine_if_rule0(text) or determine_if_rule1(text))

def parse_empty_string(): # 구문해석_빈_문자열이 들어왔을시 리턴값
    return{'node':"''",'rule':0}
def default_node_information(text, offset): # 기본 노드 정보
    res={}
    res={
        'node':text,
        'start':offset,
        'and':len(text)-1+offset
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
def update_rule1_data(text, res):       # res : result 결과?
    assert determine_if_rule1(text)     # 룰1을 적용할조건인지 결정하는 함수가 참인지 검사하고 res값반환
    find_matching_id = find_matching_pair(text, 0)
    res['rule']=1
    return res                          # 구문분석_괄호()함수의 룰1결과예시를 참조하여 res로 반환

def update_rule1_mid(text, res):        # 
    assert determine_if_rule1(text)     # 룰1을 적용할조건인지 결정하는 함수가 참인지 검사하고 res값반환
    
    return res                          # 구문분석_괄호()함수의 룰1결과예시를 참조하여 res로 반환

def update_rule2_data(text, res):
    assert determine_if_rule2(text)     # 룰2을 적용할조건인지 결정하는 함수가 참인지 검사하고 res값반환
    res['rule']=2
    return res                          # 구문분석_괄호()함수의 룰2결과예시를 참조하여 res로 반환

def update_rule2_nodes(text, res):
    assert determine_if_rule2(text)     # 룰2을 적용할조건인지 결정하는 함수가 참인지 검사하고 res값반환
    
    return res                          # 구문분석_괄호()함수의 룰2결과예시를 참조하여 res로 반환

def parse_parentheses(text): # (구문문석-pars_괄호_parentheses)
    """For the given string, parse it in the form of dict. 
       # 주어진 문자열에 대해, dict형식으로 구문분석(parsine)결과를 출력하는 함수이다.
    For detailed explanation about the parsing process and the result format, consult parentheses/documents/assignment.txt file.
    # 디테일한 설명, 파싱(구문분석) ​​과정과 결과 형식에 대해서 괄호/문서/할당.txt 파일을 참조하세요.
    Args:                                       # 독립변수들
        str text                                # text인수를 str형식으로 받는다.
    Returns:
        dict                                    # 반환값을 dict형식으로
    Raises:                                     # Raises 키워드로 예외처리 2가지 지정
        InvalidTokenException: When the input contains invalid character.       #토큰문자열이 아닌듯한
        NotClosedParenthesesException: When the input have a syntax error.      # 닫힌괄호매칭이 안될듯한
    Examples:               #예시
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
    
    return parse_parentheses_with_offset(text) # (구문문석-parse_괄호_parentheses)함수의 반환값을 이 함수(text)로

def parse_parentheses_with_offset(text, offset = 0):  # 함수(a,b=0) # offset은 뭔가 첫함수에 들어가는 인덱스관련값
    rule0 = determine_if_rule0(text)                  # 앞서 룰0,1,2결정함수에서 text에 따라 True,False반환값 대입
    rule1 = determine_if_rule1(text) 
    rule2 = determine_if_rule2(text) 

    if rule0:                                         # 룰0이 True인경우 실행
        return parse_empty_string()                   # 룰0은 공백일 경우로 empty_string()실행?값을 반환
    
    res = default_node_information(text, offset)      # 공백이 아닌 노드?텍스트의 경우 실행하여 res에 대입

    if rule1:                                         # 룰1이 True인경우 실행
        res = update_rule1_data(text, res)                # 롤1data 함수 실행 반환
        res = update_rule1_mid(text, res)                 # 룰1mid 함수 실행 반환(근데이럼미드만남는건아닌가)
    elif rule2:                                       # 룰2이 True인경우 실행
        res = update_rule2_data(text, res)                # 룰2data 함수 실행 반환
        res = update_rule2_nodes(text, res)               # 룰2mid 함수 실행 반환
    else:                                               
        assert False, 'Something goes wrong'          # assert : 거짓이면 AssertionError예외를 발생
    
    return res 

def main():                                     # C,Java와 달?리 Python에선 특별한 함수이름이 아?님.
    args = sys.argv                             # sys.argv는 sys모듈에서 제공하는 명령줄인수? 저장리스트
    with open(f'{sys.argv[1]}', 'r') as f:      # with문:파일 등 자원을 열고 자동으로 해제.
        text = f.read().strip()                 # 파일의 내용을 읽고 앞뒤 공백을 제거
        print(parse_parentheses(text))          # 읽은 내용을 parse_parentheses 함수로 처리한 결과를 출력

if __name__ == '__main__':                      # 스크립트가 직접 실행될 때만 main() 함수를 호출합니다.
    # main()  # 외부코드에서 부를때는 실행되지 않지만 이 코드파일로 돌릴때만 실행
    print(find_matching_pair('()()(())', 2))