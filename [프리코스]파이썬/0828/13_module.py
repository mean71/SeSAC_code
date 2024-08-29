# 13_module.py
# 학습 날짜 : 8월 28일 수요일
# 파이썬 프리코스 8회차
# 학습 내용 : 모듈(module)

# 함수
# 코드 뭉치를 함수 단위로 관리한다.

# 모듈
# 함수들을 파일 단위로 관리한다.
# 예시
# 더하기 함수, 빼기 함수, 나누기 함수, ..., 수학과 관련된 함수
# 수학과 관련된 함수들을 모아서 다른 파일로 관리하자 -> 모듈


# 파일(모듈)을 불러오기

# math_module 파일을 불러온다.
import math_module


# math_module에 있는 add 함수 호출
result = math_module.add(1, 2)
print(f"result = {result}")

# math_module에 있는 sub 함수 호출 코드 작성
result = math_module.sub(20, 10)
print(f"result = {result}")

# 패키지 : 파일(모듈)들의 뭉치
# django : 파이썬 웹 개발 관련 패키지
# 패키지 관리 매니저 : 패키지를 설치하거나 제거하거나 관리하는 도구
# -> PyIP(PIP)

# pip install requests
# pip3 install requests


# from 모듈명 import 함수명
# 특정 모듈에서 특정 함수만 불러오는 방법

# import 모듈명
# 특정 모듈의 모든 함수를 불러오는 방법

# math_module에서 add함수만 불러오는 코드
from math_module import add

# import를 통해 불러온 함수는 모듈명 없이 바로 호출할 수 있다.
result = add(10, 1)
print(f"result = {result}")

# as 키워드 : 모듈 또는 함수의 별칭(별명)을 지정한다.
from math_module import add as math_add

# from string_module import add

# as 키워드의 역할
# 중복되는 함수명을 피한다.
# 긴 함수명(모듈명)을 짧게 수정한다.

# import pandas as pd


# 파이썬 내장 모듈
# 모듈, 라이브러리, 패키지
# random, time

import random

# 난수를 다루는 모듈

# 0~10사이(10을 포함한) 정수 중 하나를 돌려준다.
random_number = random.randint(0, 10)
print(random_number)

# 리스트 중 하나의 원소를 랜덤으로 선택해서 돌려준다.
random_choice = random.choice([1, 2, 3, 4, 5])
print(random_choice)


import time

# 시간(초)와 관련된 모듈

# 일정 시간(초) 프로그램의 실행을 일시 정지하는 함수
# time.sleep(초)
# print("sleep 시작")
# time.sleep(5)
# print("sleep 종료")

# 간단한 타이머
# 10초

# print("타이머 시작")

# for _ in range(10):
#     time.sleep(1)

# print("타이머 종료")

# 사용자에게 시간을 입력받는 타이머
"""
N = int(input("타이머 시간을 입력하세요 : "))

print(f"{N} 초 타이머 시작")

for t in range(N):
    print(f"{t}초가 지나갔습니다.")
    time.sleep(1)

print("타이머 종료")
"""

# 현재 시간을 초로 만들어서 돌려준다.
# 1960년 1월 1일 0시 0초 부터 현재까지의 차이를 초로 변환한 숫자를 돌려준다.
second = time.time()
print(second)


# 프로그램의 실행 시간 측정

start = time.time()  # 프로그램의 실행 시간

list_ = []
for _ in range(10000000):
    list_.append(1)

end = time.time()  # 프로그램의 종료 시간
print(end - start)
