# 09_loop2.py
# 학습 날짜 : 8월 23일 금요일
# 파이썬 프리코스 5회차
# 학습 내용 : while 반복문

# while 반복문
# 조건식이 들어가는 반복문
# 조건식의 결과가 참(True)이면 반복하는 반복문

# while 반복문 사용법
"""
while 조건식:
    1 tab으로 구분한 코드 블럭
    반복해서 실행할 코드 블럭
"""

# 1부터 10까지 출력하는 반복문 코드

# 1부터 10까지 변화할 변수 number
number = 1

print(number)  # 출력 : 1
number = number + 1

print(number)  # 출력 : 2
number = number + 1

print(number)  # 출력 : 3
# number가 11이 아니면 위 코드를 반복해서 실행
# 조건식 : number != 11
print()

# 조건의 기준 변수 number
number = 1
while number != 11:
    print(number)
    number = number + 1
    """
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10

    """

# 조건식의 조건 변수 number
number = 1
while number != 11:
    print(number)
    number = number + 1  # 조건 변수 number에 대한 변화식이 필수

# while 반복문 주의할점
# 1. 조건식의 조건 변수
# 2. 조건 변수에 대한 적절한 변화식이 필수
# - 적절한 변화식 : 조건식을 언젠가는 만족하지 않게 만드는 코드


# 문제. number2에 2를 곱한 값을 반복해서 출력
# 값(number2)이 1000 이하일 때 반복하는 코드
# 출력 : 2 4 8 16 32 ...

# 조건식의 기준 변수
number2 = 2

# 조건식 : number2 <= 1000
while number2 <= 1000:
    print(number2)
    # number2 * 2한 값을 다시 number2에 할당
    number2 = number2 * 2  # 기준 변수에 대한 변화식
    """
    2
    4
    8
    16
    32
    64
    128
    256
    512
    """


# "hello"를 3번 출력하는 코드
for _ in range(0, 3):
    print("hello")
    # break : 반복문의 반복을 종료하는 키워드
    break

number = 1
while number < 10:
    print(number)  # 출력 : 1
    number = number + 1
    # break: break 아래 코드의 실행을 막는다.
    break

print(number)  # 출력 : 2


number = 1
while number < 10:
    print(number)  # 출력 : 1
    # break: break 아래 코드의 실행을 막는다.
    break
    number = number + 1
    number = number + 1
    number = number + 1

print(number)  # 출력 : 1

# 리스트 내부 원소를 하나씩 출력하는 코드
# 이 때, 10보다 크면 반복문을 종료한다.
# 이 때, 10보다 크면 출력하지 않아야 한다.
numbers3 = [0, 4, 2, 11, 3, 2, 1, 5]

for number in numbers3:
    # 출력 -> 조건 평가(검사)
    print(number)  # 결과 : 0 4 2

    if number > 10:
        # 원소(number)가 10보다 클 때 break 실행
        break

numbers3 = [0, 4, 2, 11, 3, 2, 1, 5]

for number in numbers3:
    # 조건 평가(검사) -> 출력
    if number > 10:
        # 원소(number)가 10보다 클 때 break 실행
        break
        # break와 동일 코드 블럭이면서 break 밑에 있는 코드는 실행이 절대 X

    print(number)  # 결과 : 0 4 2


# while 반복문과 break
# 문제. number3에 대해 더하기 10을 한 값을 반복 출력
# number3 = number3 + 10
number3 = 10

# 임의의 무한 반복문
while True:
    print(number3)
    number3 = number3 + 10
    # 이 때, number3이 100을 초과하면 반복문을 종료하는 코드를 작성
    # 조건식 : number3 > 100
    if number3 > 100:
        print("반복문을 종료합니다.")
        break

number3 = 10
while number3 <= 100:
    print(number3)
    number3 = number3 + 10


# continue 키워드
# 1. continue 아래 코드를 실행하지 않는다.
# 2. 다음 반복으로 넘아가게 한다.

numbers4 = [1, 2, 3, 11, 4, 5]
# 리스트 원소 중 10 이하의 값만 출력
# -> 10 초과인 원소의 반복은 넘어가게 한다.
for number in numbers4:
    if number > 10:
        print("continue를 실행합니다. 아래 코드들은 실행이 되지 않습니다.")
        continue
        print("continue 아래 코드는 실행 되지 않습니다.")
    print(number)
