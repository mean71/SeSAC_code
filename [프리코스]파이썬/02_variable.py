# 02_variable.py
# 학습 날짜 : 8월 20일 화요일
# 파이썬 프리코스 2회차
# 학습 내용 : 변수

# 변수는 무엇이냐?
# 데이터를 저장하는 상자 / 공간

# 변수 생성과 데이터(값) 저장
# 변수명 = 데이터
# = (할당/저장 연산자) : 변수(명)에 데이터를 할당(저장)한다.
# 수학 : 1 = 1, 같다
# 개발 : 할당(저장)

a = "hello world" # 문자열 "a"를 변수 a 에 할당(저장)
number = 1 # 정수 1을 변수 number에 할당
number2 = 1.0 # 실수 1.0을 변수 number2에 할당

# 그래서 변수라는 개념을 왜 사용해야하는가?

# 1. 변수는 데이터(값)에 의미를 부여
# 변수 string에 무슨 값이 있는지 모르지만
# 문자열(string)이 저장됐구나
string = "hello world" 

# 안좋은 예시
a = 1
b = "hello world"
c = 2
aa = 10
bb = 20

# 좋은 예시
number = 1
age = 20
password = "password"

# 2. 데이터의 재사용과 가독성
print("hello world hello world hello world hello world hello world")
print("hello world hello world hello world hello world hello world")
print("hello world hello world hello world hello world hello world")
print("hello world hello world hello world hello world hello world") 
print("hello world hello world hello world hello world hello world")

# 변수 string은 문자열 hello world hello world hello world hello world hello world 저장
string = "hello world hello world hello world hello world hello world" 
print(string)
print(string)
print(string)

# 3. 유지보수
print("hello python")
print("hello python")
print("hello phyton")
print("hello python")
print("hello python")

string = "hello python"
print(string)
print(string)
print(string)
print(string)
print(string)


# 파이썬 기능의 이름(예약어)
# type, int, str, bool

# 예약어는 변수의 이름으로 사용하면 안된다.
# type = "type" # 변수명 type에 문자열 "type" 저장

# 데이터의 자료형을 확인하는 기능 type()
# 기존 기능은 사라지고, 문자열 "type"만 저장
# print(type(1))


# 변수의 데이터 재할당
# 기존 변수에 새로운 데이터를 할당(저장)
number = 1 # 변수 number에 데이터 1을 할당(저장)
print(number)

number = 2 # 변수 number에 데이터 2를 할당(저장)
print(number)


# 동시 할당
# 하나의 줄에 여러 변수와 값을 생성하고, 저장하는 방법
number1 = 1
number2 = 2

# number3과 number4에 각각 데이터 3,4를 할당(저장)
number3, number4 = 3, 4

# 2개의 변수에 하나의 데이터를 저장
# 데이터 10을 변수 y에 저장하고, 변수 y를 변수 x에 저장한다. 
x = y = 10

