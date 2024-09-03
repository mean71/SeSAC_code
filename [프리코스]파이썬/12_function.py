# 12_function.py
# 학습 날짜 : 8월 27일 화요일
# 파이썬 프리코스 7회차
# 학습 내용 : 함수

# 기존에 사용해봤던 함수(도구, 기능)
# type(), int(), float(), str(), bool()
# len()

# len()을 통해 함수의 개념을 학습
# 어떠한 입력(리스트, 문자열, 튜플)을 넣으면
# 내부에서 (우리가 알 수는 없지만) 특정 동작을 수행하고
# 특정 결과(출력, 리스트의 길이 / 원소의 개수)을 돌려준다.

# type()
# 입력(모든 자료형 데이터)
# 동작(알 필요 X)
# 결과(입력으로 넣은 데이터의 자료형)

# int()
# 입력(문자열, 실수형)
# 동작
# 결과(정수형으로 변환한 데이터)

# 파이썬 내장 함수
# 파이썬을 만든 개발자가 미리 만들어 놓은 함수

# 파이썬 사용자 정의 함수
# 개발자가 직접 만든 함수

# 함수 정의 방법
"""
def 함수명(매개변수):
    1 tab으로 구분한 코드 블럭
    함수 내부에서 실행할 코드
    
    함수를 실행(호출)한 위치에 돌려줄 값을 지정
    함수의 결과 값을 지정
    return 값
"""
# int() 예시
"""
def int(매개변수):
    입력받은 값을 정수형으로 변환하는 무수히 많은 코드
    ...
    ...
    ...
    
    반환값
    결과값
    값을 돌려준다
    return 정수형으로 변환한 값

string_1 = "1"
int(string_1)
# 다른 자료형을 정수형으로 변환 후 데이터 생성(돌려준다, 반환하다)
"""

# 매개변수란?
"""
for 내부변수 in [1,2,3]:

for _ in range(0,10):
"""

"""
def 함수명(매개변수):
    매개변수 : 함수 내부에서 입력받은 값을 저장할 변수

def 함수명():
    매개변수가 없는 함수
"""
# 매개변수가 없는 함수
# 1+1 을 돌려주는(반환하는) 함수
def func_1():
    result = 1 + 1
    return result

# 함수의 호출(call)
# 함수를 실행할 때 입력해야할 값(매개변수)이 없어서
# 소괄호() 안에 값을 넣지 않는다.
call_1 = func_1()
print(call_1)

# 매개변수가 있는 함수
# 입력 받은 값에 + 1 해주는 함수

def func_2(number):
    result = number + 1
    return result

# 함수의 호출(call)
# 변수 call_2에는 할당이 된다.
# 정수 10을 전달한 func_2 호출 결과값이 할당이 된다.
call_2 = func_2(10)
print(call_2)

def plus_one(number):
    result = number + 1
    return result

call_3 = plus_one(2)
print(call_3)

# 문제
# 매개변수가 있는 함수
# 함수 이름은 func_3
# 내부 동작은 아래와 같다.
# 입력 받은 값 + 1
# string = f"더하기 1 결과는 {입력 받은 값 + 1}"
# 변수 string 을 결과 값으로 돌려준다.

def func_3(number):
    result = number + 1
    string = f"더하기 1 결과는 {result}"
    return string

call_4 = func_3(20)
print(call_4)

call_5 = func_3(10)
print(call_5)
# 인자 / 인자값 / argument
# 함수에 전달하는(입력하는) 값
"""
def 함수명(매개변수1, 매개변수2, 매개변수3, 매개변수4, ...):
    함수 내부 동작 코드 블럭
    return 함수를 호출한 위치에 돌려줄 값
# 함수 호출
함수명(인자1, 입력값2, 입력값3, ...)
"""
# 문제
# 매개변수 2개인 함수
# 매개변수 1의 이름 : number1
# 매개변수 2의 이름 : number2
# 함수의 결과 값 : number1 + number2
def func4(number1, number2):
    result = number1 + number2
    return result

# 함수의 호출 코드
# 함수 인자 값으로 정수 1과 정수2를 전달
call_6 = func4(1, 2)
print(call_6)

call_7 = func4(10, 20)
print(call_7)

# 문제
# 매개변수가 2개인 함수
# 매개변수1의 이름은 x
# 매개변수2의 이름은 y
# 내부 동작은 아래와 같습니다.
# 1. 만약 x가 크다면 결과값으로 x를 돌려준다.
# 2. 만약 y가 크다면 결과값으로 y를 돌려준다.
# 3. 만약 x와 y가 같다면 문자열 "같다"를 돌려준다.

def func_5(x, y):
    if x > y:
        # 코드 블럭의 구분
        # return x 코드는 x > y 조건 아래에서 실행한다.
        return x
    if y > x:
        return y
    if x == y:
        return "같다"

call_8 = func_5(2, 1)
print(f"call_8에 저장된 값 : {call_8}")
call_9 = func_5(2, 3)
print(f"call_9 = {call_9}")
call_10 = func_5(2, 2)
print(f"call_10 = {call_10}")

def func_6(x, y):
    if x > y:
        return x
    else: return y

# len() : 컨테이너 자료형의 길이를 반환해주는 함수
# len()을 직접 구현

def len(container):
    # 어떻게 하면 컨테이너 자료형의 원소의 개수를 카운트할 수 있을까?
    # 원소의 개수만큼 반복하는 반복문 코드를 작성
    # 원소의 개수 만큼 1씩 증가할 변수
    result = 0

    for element in container:
        print(element)  # 컨테이너 내부 원소를 출력
        # 반복문 내부에서 1 만큼 증가하는 코드
        result = result + 1
        # return이 반복문 내부에 있다면
        # 반복이 1번만 일어나고 함수(반복문) 종료

    # 원소의 개수만큼 반복해서 더하기 1을 한 값을 돌려준다.
    return result

list_1 = [3, 2, 5, 1, 2]
length = len(list_1)
print(f"list_1의 길이는 {length}")

string_1 = "hello world"
length_2 = len(string_1)
print(f"string_1의 길이는 {length_2}")