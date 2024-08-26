# 07_string.py
# 학습 날짜 : 8월 22일 수요일
# 파이썬 프리코스 4회차
# 학습 내용 : 문자열 자료형

# 문자열의 특징
# 1. 문자들의 나열
# 2. 0개 이상의 문자를 저장할 수 있는 컨테이너 자료형
# 3. 수정(문자의 추가, 삭제, 재할당)이 불가능하다.

# 문자열 연산(+ *)
# 리스트와 동일하다.

# 문자열 인덱싱
# 리스트와 동일하다.
string = "abc"
# b 만 선택
word = string[1]
print(word)  # 출력 : b

# 문자열 슬라이싱
# 리스트와 동일하다.

# 수정이 불가능하다.
# 리스트 원소의 수정 -> 리스트[위치] = 값
string = "abc"
# string[0] = "A" # 오류가 발생

# 문자열의 추가(?)
# 새로운 문자열을 생성하는 방법, 추가와는 동작 방식이 다르다.
string = string + "d"
print(string)  # 출력 : abcd


# f-string : 문자열 포매팅
# 문자열 포매팅 : 문자열 내부에 변수 또는 데이터를 삽입하는 방법
# 자기소개 출력 코드
name = "정우영"
print("저의 이름은", name, "입니다.")  # f-string이 없을 때

print(f"저의 이름은 {name} 입니다.")  # f-string이 있을 때

print(f"1 + 1 = {1 + 1}")
print(f"1 > 0 = {1 > 0}")

# 튜플(tuple) 자료형
# 1. 자료형 상관없이 0개 이상의 데이터를 저장할 수 있다.(리스트)
# 2. 수정(추가, 삭제, 원소 재할당)이 불가능하다.(문자열)

# 튜플 생성 방법 : 소괄호 ()
number_tuple = (1, 2, 3, 4)
print(number_tuple)
print(type(number_tuple))  # 출력 : <class 'tuple'>

number_tuple2 = 1, 2, 3, 4  # 패킹 문법
print(number_tuple2)
print(type(number_tuple2))  # 출력 : <class 'tuple'>
