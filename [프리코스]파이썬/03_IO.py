# 03_IO.py
# 학습 날짜 : 8월 20일 화요일
# 파이썬 프리코스 2회차
# 학습 내용 : 사용자 입력과 출력

# 출력(print)

# print() 내부에서 데이터 구분(,)
# 쉼표(,)를 기준으로 공백으로 나눠서 데이터를 출력
print(1,2,3,4,5,6,7,8)

word1 = "hello" 
word2 = "world"
# 변수를 , 기준으로 출력
print(word1, word2)

# end 옵션
# 출력 결과의 마지막 문자를 결정하는 옵션
# 기본값 : 줄바꿈(개행 문자)

print(1,end=" + ")
print(2)

# 사용자 입력(input)
# input("프롬프트(안내 메시지)") : 사용자가 입력할 수 있게 도와주는 기능
# 이름과 이메일
name = input("이름을 입력해주세요 : ") # 입력받은 데이터를 변수 name에 할당(저장)
email = input("이메일을 입력해주세요 : ") # 입력받은 데이터를 변수 eamil에 할당(저장)

# 입력받은 데이터를 각각 출력
print(name)
print(email)

# number = input() # 프롬프트(안내 메세지) 없이 input() 기능을 사용할 수 있다.
number = input("숫자를 입력해주세요. : ")

# 만약 숫자를 입력받았다면 이 데이터의 자료형은 무엇인가?
# input() 기능을 통해 입력 받은 데이터 자료형은 모두 문자열(str)이다.
print(number)
print(type(number))

# 입력받은 숫자 모양의 데이터를 숫자형 데이터로 사용하고 싶으면 어떻게 해야할까?
# 형 변환, 문자열(str) -> 정수형(int) / 실수형(float)
print(int(number)) # 정수형으로 변환
print(float(number)) # 실수형으로 변환

# input() : 할당(X)
# 사용자에게 데이터를 입력 받는 도구