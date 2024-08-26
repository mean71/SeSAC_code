# 05_condition.py
# 학습 날짜 : 8월 21일 수요일
# 파이썬 프리코스 3회차
# 학습 내용 : 조건문

# 조건문 : 조건식의 결과(참/거짓)에 따라 실행할 코드를 제어하는 문법
# 조건문의 작성법
# if 키워드 활용
"""
if 조건식:(콜론, shift + l 오른쪽 키)
    코드 블럭, 조건식이 참(True)일 때 실행되는 코드 뭉치
    코드 블럭의 표현은 1 Tab 
"""
# 변수 number의 값이 15보다 크면 O를 출력
number = 20
if number > 15:  # 콜론(shift + L 오른쪽 키(;))
    print("O")


# 변수 number2가 1과 같다면 number2의 데이터를 출력하는 코드 작성
number2 = 1
if number2 == 1:  # 조건식이 참(True)이면 내부 코드 블럭을 실행
    print(number2)


# 거짓(False)이면 어떤 코드를 실행? -> else 키워드
number2 = 2
# number2가 1과 같다면 number2를 출력 아니면 X 를 출력
if number2 == 1:
    print(number2)
else:  # number2가 1과 같지 않으면 X 출력
    print("X")

# 연습
number3 = 49
# number3 곱하기 2가 100 이상이면 "number3 * 2는 100이상이다" 를 출력
# 아니라면 "number3 * 2는 100미만이다" 를 출력
# 조건식 : number3 * 2 >= 100
if number3 * 2 >= 100:
    print("number3 * 2는 100이상이다")
else:
    print("number3 * 2는 100미만이다")

# 여러개의 조건식
# elif 조건식
# if와 함께 사용해야한다.
number4 = 25
# 문제, 한 번의 출력만 해야한다.
# number4가 30이상이면 "30이상이다", number4 >= 30
# number4가 25이상이면 "25이상이다", number4 >= 25
# number4가 20이상이면 "20이상이다", number4 >= 20
if number4 >= 30:
    print("30 이상이다")
elif number4 >= 25:
    # 위의 조건식이 모두 거짓이면 조건식을 확인 O
    # 위의 조건식 중 하나라도 참이면 확인 X
    print("25 이상이다")
elif number4 >= 20:
    print("20 이상이다")

# if를 여러개 사용한 방법 Vs. if와 elif를 여러개 사용한 방법
# if를 여러개 사용한 방법은 모든 조건식을 확인
# if와 elif를 여러개 사용한 방법은 하나의 조건식이 참이면 나머지 조건식은 실행 자체 X

score = 70
# 문제
# 1. score가 90보다 크면 "A"를 출력
if score > 90:
    print("A")
# 2. score가 80보다 크면 "B"를 출력
elif score > 80:
    print("B")
# 3. score가 70보다 크면 "C"를 출력
elif score > 70:
    print("C")
# 4. score가 아니라면 "F"를 출력
else:
    print("F")
