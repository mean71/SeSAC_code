# 09_loop.py
# 학습 날짜 : 8월 22일 목요일
# 파이썬 프리코스 4회차
# 학습 내용 : 반복문

# for 반복문 사용법
"""
내부 변수란 컨테이너 자료형의 내부 원소
for 내부 변수 in 컨테이너 자료형 데이터(변수):
    1tab으로 구분한 코드 블럭
"""
# 리스트 활용 반복문
list1 = [1, 1.0, "1"]

# list1 내부에 저장된 원소들이 variable에 하나씩 할당되면서 반복 수행한다.
for variable in list1:
    print(variable)
    # 출력
    # 1
    # 1.0
    # "1"

print()
# 문제. 반복문을 사용해서 각 원소에 1을 더한 값을 출력
list2 = [1, 2, 3, 4, 5]
for number in list2:
    # 실행할 코드 블럭
    result = number + 1
    print(result)
