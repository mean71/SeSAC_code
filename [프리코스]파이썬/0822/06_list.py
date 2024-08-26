# 06_list.py
# 학습 날짜 : 8월 21일 수요일
# 파이썬 프리코스 3회차
# 학습 내용 : 리스트 자료형

# 리스트 자료형 특징
# 1. 하나의 변수에 0개 이상의 데이터를 저장
# 2. 자료형 상관없이 데이터를 저장할 수 있다.

# 리스트 자료형 생성 방법 : 대괄호 []
# number = 1, number = 1.0
list1 = []  # 빈 리스트를 생성하고, 변수 list1에 할당(저장)
list2 = [1]  # 숫자 1을 저장한 리스트를 생성하고, 변수에 할당

# 여러개의 데이터를 저장한 리스트를 생성하고, 변수에 할당
# 여러개의 데이터를 구분하는 방법 : 쉼표(,)
list3 = [1, 2, 3, 4, 5, 6]

# 정수 1, 실수 1.0, 문자열 "1"을 저장한 리스트를 생성하고, 변수에 할당하는 코드
list4 = [1, 1.0, "1"]
print(list4)

# type() : 데이터(변수)의 자료형을 확인하는 도구(기능)
print(type(list4))

# 리스트에 저장된 데이터(값) -> 원소(요소)

# len() : 리스트에 저장된 데이터(원소) 개수 확인하는 도구(기능)
print(len(list4))

# 리스트의 연산자(+, *)
# 더하기(+) : 두 개의 리스트를 합쳐서 새로운 리스트를 생성
list5 = [1, 2]  # 1,2가 저장된 리스트를 생성하고, 변수 list5에 할당
list6 = [3, 4]

# list5와 list6을 합쳐서 생성된 리스트를 변수 list7에 할당(저장)
list7 = list5 + list6
print(list7)

# 곱하기(*) : 하나의 리스트를 곱한 수 만큼 반복한 리스트를 생성
# list7에 곱하기 5를 한 결과 리스트를 변수 list8에 할당
list8 = list7 * 5
print(list8)


# 리스트 인덱싱(indexing)
# 위치(index)를 사용해서 리스트 내부 원소(데이터)에 접근하는 방법

# 1, 2, 3, 4, 5가 저장된 리스트를 생성하고, 변수 number_list에 할당
number_list = [1, 2, 3, 4, 5]

# 첫 번째 위치(index) 원소를 출력
# 인덱싱 방법 : 대괄호 [위치(index)]
# 첫 번째 위치는 0을 의미한다.
print(number_list[0])

# 변수 number_list의 세 번째 위치의 원소를 출력하는 코드 작성
print(number_list[2])

# 음수 인덱싱(-1, -2, -3, ...)
# 끝에서 부터 원소를 찾는다.

# 마지막 위치의 원소를 출력
print(number_list[-1])


# 인덱싱을 통한 원소(데이터)의 수정(재할당)
"""
number = 1
number = 2
"""
# 세 번째 원소의 값을 10으로 수정
# number_list = [1, 2, 3, 4, 5]
# 위치(인덱스)    0, 1, 2, 3, 4
number_list[2] = 10
print(number_list)


# 슬라이싱
# 리스트의 특정 구간을 분할
number_list = [1, 2, 3, 4, 5]

# 슬라이싱 사용법
# 리스트[시작위치:마지막위치:간격]
# 마지막 위치의 데이터(원소) 포함 되지 않는다.

# 첫 번째부터 세 번째까지 분할해서 리스트를 생성 후 변수에 할당
# 간격 값은 생략 -> 간격은 1이 된다.
slice1 = number_list[0:3]
print(slice1)

# 첫 번째부터 네 번째까지 분할, 간격을 2
# (1), 2, (3), 4, 5
slice2 = number_list[0:4:2]
print(slice2)  # 결과 : [1, 3]


# 음수 인덱싱을 활용한 슬라이싱
# 뒤에서 세 번째부터 뒤에서 첫 번째까지 슬라이싱
# -3 -> -1
# number_list = [1, 2, 3, 4, 5]
slice3 = number_list[-3:-1:1]
print(slice3)  # 결과 : [3, 4]

# 마지막 위치 값을 생략하는 경우
# 시작 위치부터 마지막 위치의 원소까지 분할한다.
slice4 = number_list[-3::1]
print(slice4)  # 결과 : [3, 4, 5]

# 슬라이싱 연습
string_list = ["a", "b", "c", "d", "e", "f", "g"]

# 문제 1. a b c 가 저장된 리스트를 분할(슬라이싱)을 통해 생성 후 출력
slice5 = string_list[0:3]
print(slice5)
# 문제 2. c d e f 가 저장된 리스트를 분할을 통해 생성 후 출력
slice6 = string_list[2:6]
print(slice6)
# 문제 3. e f g 가 저장된 리스트를 분할을 통해 생성 후 출력
slice7 = string_list[4:]
print(slice7)
# 문제 4. a c e 가 저장된 리스트를 분할을 통해 생성 후 출력
slice8 = string_list[0:5:2]
print(slice8)

# 리스트 원소의 추가와 삭제
# [1,2,3,4] / 리스트 슬라이싱

number_list = [1, 2, 3, 4, 5]
# 리스트변수.append(데이터) : 리스트에 데이터(원소)를 마지막에 추가하는 도구(기능)
# number_list에 원소 6을 추가
number_list.append(6)
print(number_list)

# 리스트변수.pop() : 리스트 마지막 원소의 삭제 도구
number_list.pop()
print(number_list)

# pop() 두 가지 특징
# 1. 삭제할 위치를 지정
# 2. 리스트에서 삭제한 원소를 데이터로 생성(돌려준다)

# 리스트변수.pop(위치) : 삭제할 위치를 지정
# 첫 번째 위치의 원소를 삭제
number_list.pop(0)
print(number_list)  # 출력 : [2, 3, 4, 5]

# 원소 숫자 3을 삭제
number_list.pop(1)
print(number_list)  # 출력 : [2, 4, 5]

# 삭제한 원소를 데이터로 돌려준다.

# 원소 5가 리스트에서 삭제되면서 데이터 5를 돌려준다.
number = number_list.pop()
print(number)  # 출력 : 5
