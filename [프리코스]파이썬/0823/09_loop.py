# 09_loop.py
# 학습 날짜 : 8월 22일 목요일
# 파이썬 프리코스 4회차
# 학습 내용 : for 반복문

# for 반복문 사용법
"""
내부 변수란 컨테이너 자료형의 내부 원소
for 내부 변수 in 컨테이너 자료형 데이터(변수):
    1 tab으로 구분한 코드 블럭
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


# 문제. 문자열 형태의 정수가 저장된 리스트를 반복해서
# 각 원소를 정수형(int)로 변환해서 출력(type())
# 정수형 원소가 저장된 새로운 리스트를 생성해서 출력하세요.
list3 = ["1", "2", "3"]

# 정수로 변환한 데이터를 저장할 빈 리스트를 생성 후 변수에 할당
list4 = []

for string in list3:
    # 변수 string의 자료형을 정수형(int) 변환 후 변수에 할당
    number = int(string)
    # print(number, type(number))
    list4.append(number)
    print(list4)


# 레인지(range)의 for 반복문
# 리스트와 사용법은 동일
# 1부터 10까지의 연속된 정수 목록을 출력
for number in range(1, 11, 1):
    print(number, end=" ")  # 1 2 3 4 5 6 7 8 9 10

print()
# 문제. 2부터 20까지 연속된 정수 목록을 출력, 짝수만 출력
# range()만 활용할 것.
# 2 4 6 8 10 ... 20
for number in range(2, 21, 2):
    print(number, end=" ")  # 출력 : 2 4 6 8 10 12 14 16 18 20
print()

# N(특정 수)만큼 반복하고 싶은 코드가 있을 때
# range 활용
# 5번 반복하는 코드
for number in range(5):
    print("안녕하세요.")

# 반복문 내부 변수 생략하는 방법
# 언더바(_) : 생략 의미
for _ in range(5):
    print("안녕하세요.2")

# 리스트 인덱스(위치) 활용
# 1. 리스트의 범위
# 리스트: [1, 2, 3, 4]
# 위치:    0, 1, 2, 3 -> 0 부터 3 까지의 연속된 정수 목록
# len(리스트) : 리스트의 길이(원소의 개수) 구해주는 도구(기능)

# 리스트의 끝 위치(index) -> 리스트의 길이 - 1
number_list = [1, 2, 3, 4, 5, 6, 7]

# 리스트 길이의 결과를 변수 end 할당
end = len(number_list)

# 연속된 위치(index) 목록을 반복해서 출력
for index in range(0, end):
    # print(index)
    # 위치(index)를 활용해서 원소에 접근 후 출력
    print(number_list[index])


# 문제. 위치(index)가 3 이상인 원소만 출력하는 코드 작성
# 출력 : 7 2 1 9 10
number_list2 = [0, 4, 1, 7, 2, 1, 9, 10]
print()
# 두 가지 방법
# 첫 번째 range 시작 정수를 3으로 시작
end = len(number_list2)
for index2 in range(3, end):
    print(number_list2[index2])
print()

# 두 번째 조건문을 활용하는 방법
# 연속된 위치(index) 목록의 끝 정수 end
end = len(number_list2)
for index3 in range(0, end):
    # 조건문, 조건식이 위치(index) >= 3
    # 조건식이 참이면 해당 위치(index)의 원소 출력
    if index3 >= 3:
        print(index3, number_list2[index3])
