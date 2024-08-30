# 08_range.py
# 학습 날짜 : 8월 22일 목요일
# 파이썬 프리코스 4회차
# 학습 내용 : 레인지(range)

# 레인지(range)
# 연속된 정수 목록을 만드는(저장한) 도구(컨테이너 자료형)
# 1 2 3 4 5
# 1 3 5 7 9
# 2 3 4 5
# range 사용법
# range(시작 정수, 끝 정수, 간격)
# 1 부터 10까지의 연속된 정수 목록 생성
range1 = range(1, 11, 1)  # 출력 : range(1, 11)
print(range1)

# 0 부터 5까지 2씩 증가하는 정수 목록 생성
range2 = range(0, 6, 2)  # 출력 : range(0, 6, 2)
print(range2)

# range 원소 확인 : 리스트(list) 형 변환
list1 = list(range1)
print(list1)  # 출력 : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

list2 = list(range2)  # range(0, 6, 2)
print(list2)  # 출력 : [0, 2, 4]

# 정수 목록을 생성, 실수는 사용 X
# range3 = range(0, 10.5, 0.5)  # 오류 발생, 실수를 사용해서

# 시작 정수와 간격을 생략, 끝 정수만 작성
range4 = range(10)  # range(0, 10, 1)

# 다른 컨테이너 자료형 -> 리스트 형 변환
# range -> list : 범위에 해당하는 정수 목록을 저장한 리스트를 생성

# str -> list : 각 문자를 원소로 분리해서 저장한 리스트를 생성
string = "hello   world"
string_list = list(string)

# 출력 : ['h', 'e', 'l', 'l', 'o', ' ', ' ', ' ', 'w', 'o', 'r', 'l', 'd']
print(string_list)

print(str([1, 2, 3, 4, 5]))  # 결과 : "[1, 2, 3, 4, 5]"
# 기대하는 결과 "12345" / "1 2 3 4 5"

# 멤버십 연산자
# 컨테이너 자료형 데이터(변수)에 특정 값이 포함되어 있는지 검사하는 연산자

market = ["사과", "바나나", "수박", "복숭아"]
# market에 사과가 있는지 확인(검사)
# 값 in 컨테이너
result = "사과" in market
print(result)  # 출력 : True

# market에 딸기가 없는지 확인
# 값 not in 컨테이너
result2 = "딸기" not in market
print(result2)  # 출력 : True

"""
개발자 : 여보 나 지금 퇴근. 집에 가는 길에 마트 들를건데 뭐 사다 줄까?
아내 : 우유 두 개 사와.
개발자 : 그리고?
아내 : 만약 마트에 달걀이 있으면 여섯 개 사다 줘.
"""
market = ["우유", "달걀"]

# 개발자의 관점
if "달걀" in market:
    print("우유를 여섯개 산다.")
else:
    print("우유를 2개 산다.")

# 아내의 관점
if "달걀" in market:
    print("우유를 2개 산다.")
    print("달걀을 6개 산다.")
else:
    print("우유만 2개 산다.")