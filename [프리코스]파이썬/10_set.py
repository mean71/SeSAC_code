# 10_set.py
# 학습 날짜 : 8월 26일 월요일
# 파이썬 프리코스 6회차
# 학습 내용 : 집합(set) 자료형

# 집합 자료형 특징
# 1. 중복 데이터를 저장할 수 없다.
# 2. 데이터간 순서가 없다.

# 중복 데이터를 저장할 수 없다?
set_1 = {1, 2, 3, 4, 5}

# set_1을 출력하고, 자료형을 출력하는 코드 작성
print(set_1)
print(type(set_1))

# 중복 데이터를 저장한 집합(set)
set_2 = {1, 1, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, "1", "1", "1"}

# set_2 출력 코드 작성
print(set_2)

# 정수형 1과 실수형 1.0은 같은가? 결과는 같다(True)
# print(1 == 1.0)

# 데이터간 순서가 없다?
# -> 데이터의 위치(index)가 없다
# 집합 자료형은 인덱싱과 슬라이싱이 불가능하다.
# 리스트에서 인덱싱과 슬라이싱이 가능한 이유 -> 위치가 있기 때문 -> 데이터간 순서가 있기 때문

# 집합 자료형 원소의 추가와 삭제
# 집합.add(값) : 집합 자료형에 값을 추가하는 도구

# 변수 set_3 을 생성하고, 집합 자료형을 할당
# 이 때, 집합 자료형에는 정수 1이 저장된 코드를 작성
set_3 = {1}

# add() 도구를 활용해서 정수 2를 추가하는 코드를 작성
set_3.add(2)
print(set_3)

# 집합.remove(값) : 집합 자료형에서 값에 해당하는 원소 삭제하는 도구
# 집합.discard(값): 집합 자료형에서 값에 해당하는 원소 삭제하는 도구

# 변수 set_3 에서 원소 2를 삭제를 하는 코드 작성
# remove와 discard를 한 번씩 활용하는 코드 작성 

# remove -> discard
print("remove -> discard")
set_3.remove(2)
set_3.discard(2)

set_3.add(2)
# discard -> remove
# print("discard -> remove")
# set_3.discard(2)
# set_3.remove(2)

# remove는 삭제할 원소가 없으면 오류를 발생 O
# discard는 삭제할 원소가 없어도 오류를 발생 X

# 수학 집합
# 합집합 / 차집합 / 교집합 연산
# 합집합 연산자 |(\ 원화 특수문자)
set_4 = {4, 5}
set_5 = {5, 6}
union_set = set_4 | set_5
print(union_set)  # 출력 : {4, 5, 6}

# 차집합 연산자 -
diff_set = set_4 - set_5
print(diff_set)

# 교집합 연산자 & (7 특수문자)
inter_set = set_4 & set_5
print(inter_set)

# 집합의 생성
# 빈 집합을 생성?
# 비어있는 중괄호 {} 는 딕셔너리(dict) 자료형을 생성
set_5 = {}
print(type(set_5))

# 빈 집합 생성
# set(): 다른 자료형을 집합 자료형으로 변환
# set(): 비어있는 집합 자료형을 생성하는 도구
set_6 = set()
print(type(set_6))
