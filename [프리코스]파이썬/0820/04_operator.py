# 04_operator.py
# 학습 날짜 : 8월 20일 화요일
# 파이썬 프리코스 2회차
# 학습 내용 : 연산자

# 데이터를 다루는 법(연산자)
# 산술(수학) 연산자
# 사칙연사 + - *(shift + 8) /
# 2개의 변수를 만들고, 2개의 정수 데이터를 할당(저장)
number1 = 10
number2 = 20

# number1과 number2를 더한 데이터(값)를 변수 plus 할당(저장)
plus = number1 + number2  # 10과 20을 더한 데이터를 plus 할당
print(plus)

# 빼기
minus = number1 - number2
print(minus)

# 곱하기
multiple = number1 * number2
print(multiple)

# 나누기
division = number1 / number2
print(division)

# 추가 산술 연산
# 몫 계산, 나머지 계산, 제곱 계산
# // , %(shift + 5), **

# number1을 number2로 나눈 몫
quotient = number1 // number2
print(quotient)

# number2를 number1로 나눈 나머지
remain = number2 % number1
print(remain)

# number1에 대한 2의 제곱
square = number1**2
print(square)

# 0으로는 나눌 수 없다.
# print(1%0)

# 복합 연산자(축약 연산자)
# 산술 연산과 할당 연산을 축약

# number1과 number2를 더한 데이터를 number1에 재할당
number1 = number1 + number2

# 축약 연산자 표현
number1 += number2

# 비교 연산자
# 두 데이터를 비교(같다, 다르다, 초과, 미만, 이상, 이하)한다.
# 비교의 결과는 boolean형(참 / 거짓)이다.


# 같다 ==
# 다르다 !=
# 초과, 미만 < >
# 이상(초과 또는 같다), 이하(미만 또는 같다) >= <=

number3 = 3
number4 = 4

# 같냐?
print(number3 == number4)
# 다르냐?
print(number3 != number4)
# number3은 number4 초과이냐?
print(number3 > number4)
# number3은 number4 미만이냐?
print(number3 < number4)
# number3은 number4 이상이냐?
print(number3 >= number4)
# number3은 number4 이하이냐?
print(number3 <= number4)

# 정수형 1과 문자열 "1"은 같냐? -> 거짓(False)
print(1 == "1")
print(1 == 1.0)

# 논리 연산자
# and / or : 두 개의 데이터의 불린형(참 / 거짓)결과에 따라서 새로운 불린형 데이터를 생성
# A and B
# A와 B의 불린형 결과(참 / 거짓)가 둘 다 참이면 참(True) 데이터를 생성
# 하나라도 거짓(False)라면 거짓(False) 데이터를 생성
# 거짓으로 변환하는 데이터 -> 0, "", 0.0, False
logical1 = bool(0) and bool(1)  # False and True -> False
print(logical1)

logical2 = bool(0.0) and bool(1.0)  # False and True -> False
print("logical2 =", logical2)

# and 연산자를 사용해서 참(True) 데이터 생성하는 코드 작성
logical3 = bool(1) and bool(1.0)  # True and True -> True(참) 생성, 변수에 할당(저장)
print("logical3 =", logical3)

# or : 하나라도 참(True)이면 참(True) 데이터를 생성하는 연산자
logical4 = bool(0) or bool(1)  # False or True -> True 생성
print("logical4 =", logical4)

# 참이 하나도 없으면 거짓(False) 데이터를 생성한다.
logical5 = bool(0) or bool(0.0)  # False or False -> False
print(logical5)

# not A : 하나의 데이터를 반대 불린형(참 <-> 거짓)데이터를 생성하는 연산자
logical6 = not 0  # not False -> True
print("logical6 =", logical6)

logical7 = not 1  # not 2 not 10 not 100, ...
print("logical7 =", logical7)


# 비교 연산자와 논리 연산자를 함께 활용
# 비교 연산자 : 두 개의 데이터를 비교해서 불린형(참/거짓) 데이터를 생성
# 논리 연산자 : 두 개의 불린형(참/거짓)데이터로 새로운 불린형 데이터를 생성

number1 = 10
number2 = 5
number3 = 15
# number1은 number2보다 크고, number1은 number3보다 크면 참(True) 생성
# and 연산자 : 두 개의 조건이 모두 만족
con1 = number1 > number2  # number1은 number2보다 큰가? True
con2 = number1 > number3  # number1은 number3보다 큰가? False
result = con1 and con2  # con1과 con2가 모두 참인지? True and False->False
print("result =", result)

"""
# 확장 프로그램
# 코드의 가독성을 높여주는 코드 포매터를 설치
# 코드 포매터 : 코드를 규칙에 맞게 수정 도구

# 설치 순서
# 1. 확장 탭(ctrl + shift + x)을 연다
# 2. 검색창에 black 검색
# 3. black formatter(마이크로소프트 인증마크가 붙어있는) 설치
# 4. ctrl + , (설정창)
# 5. 오른쪽 상단의 json 설정 열기 버튼 클릭
# 6. settings.json 두번째 줄에 노션 첫 화면 설정 코드를 붙여넣기
# 7. ctrl + s (저장)
print(1 == 1)
"""
