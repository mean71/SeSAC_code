# 15_object.py
# 학습 날짜 : 8월 29일 목요일
# 파이썬 프리코스 9회차
# 학습 내용 : 객체 지향 프로그래밍

# 객체
# 정보와 행동을 가진 단위(무언가)

# 리스트(객체)
# 정보 : 0개 이상의 자료형 구분없는 데이터
# 행동 : 원소를 추가, 삭제, 수정, 합치거나, 곱하거나

# 딕셔너리(객체)
# 정보 : key - value 쌍을 가진 데이터
# 행동 : 원소를 추가, 수정, 삭제

# 문자열, 튜플, 셋, 정수형, 실수형 -> 모두 객체

# 파이썬은 객체 지향 프로그래밍 언어이다.
# 파이썬에서는 객체란 곧 자료형

# 현실에 있는 객체(무언가)를 개발 세계로 가져오기 위한 개념
# 현실 객체 정우영
# 정보 : 이름, 키, 몸무게, 사는곳, 태어난곳, ...
# 행동 : 잔다, 걷다, 뛰다, 코딩을 하다, 강의를 하다, ...

# 객체 지향 이전에는 절차(행동) 지향
# C 언어 객체 개념이 X

# 네이버 서비스에 생성하다 행동(절차)
# 현실 : 메일(글) 생성하다, 블로그 글을 생성하다, 카페 글을 생성하다, ...
# 절차 지향 코드 : 생성하다(메일), 생성하다(블로그), 생성하다(카페글)

# 객체 지향 프로그래밍
# 메일을 생성하다, 블로그를 생성하다, 카페글을 생성하다.

# 객체란 정보와 행동을 가진 데이터(단위)
# 정보와 행동을 가졌으면 객체다.
# 리스트 / 딕셔너리 / 문자열 / 기타 다른 자료형

# 정우영
# 정보 : 이름 , 키 , 몸무게 ,...
# 행동 : 잔다, 걷다, 뛰다, 먹다,...

# 객체의 (정보와 행동에 대한) 설계도
# 클래스(class)

class Person:
    # 1. 생성자 메서드
    # class를 통해 객체를 생성할 때 무조건 자동으로 실행되는 메서드
    # 생성자 메서드의 이름은 __init__ 여야한다. 무조건
    # 첫 번째 매개변수로 self 가 위치한다. 무조건
    def __init__(
        self,
        init_name,
        age,
    ):
        # self란? 객체(자신)
        # 클래스를 통해 생성되는 객체를 의미
        # 클래스 내부에서 객체에 접근하기 위해서 만들어진 키워드

        # Person 객체 이름 속성(정보)
        # 저장을 받는 변수는 객체(self) 속성 = 저장 해야하는 데이터(입력으로 전달받은 매개변수)
        self.name = init_name
        # Person 객체 나이(age) 속성
        self.age = age

# 클래스(설계도)를 통해 객체를 생성
# 집합(set) 생성 방법 : set() 과 동일, 즉 클래스이름으로 객체를 생성한다.
person_1 = Person("정우영", 20)
# 마침표(.)를 통해서 속성(정보)에 접근
print(person_1.name, person_1.age)

person_2 = Person("아이유", 20)
print(person_2.name, person_2.age)

# 메서드 메서드 메서드 메서드????
# 특정 클래스가 가진 함수(행동,도구,기능)

# 함수 Vs. 메서드
# 함수는 객체(자료형) 구분없이 사용할 수 있다.
# 메서드는 객체(자료형) 종속된 함수이다.

# int(데이터) 함수 : 다른 자료형을 정수형으로 변환하는 함수
# int(리스트) int(문자열) int(실수형) int(딕셔너리) int(Person)

# 리스트.append(데이터) : 데이터를 리스트의 원소로 추가하는 메서드(행동)
# 리스트.pop()
# 딕셔너리.values()


class Person:
    # 클래스 변수(속성, 정보)
    # 해당 클래스에 의해 생성된 객체가 공유하는 속성
    species = "호모 사피엔스"

    def __init__(self, init_name, age):
        self.name = init_name
        self.age = age

    # 2. 인스턴스 메서드
    # 객체(인스턴스) 메서드 -> 객체의 행동(함수)
    # 시작 매개변수는 무조건 self이다.
    def introduce(self, start):
        print(f"{start} 저의 이름은 {self.name}이고, 나이는 {self.age}살 입니다.")

    # 생일 메서드 : 객체 age가 1 증가 하는 함수
    def birthday(self):
        self.age = self.age + 1

    # 개명 메서드 : 객체 name을 입력받은 name으로 변경하는 함수
    def rename(self, name):
        self.name = name


# Person 객체를 생성
person_3 = Person("Beemo", 20)

# Person 객체의 introduce 메서드를 실행
# 리스트변수.append()
person_3.introduce("반갑습니다.")
person_3.introduce("처음뵙겠습니다.")
person_3.introduce("안녕하세요.")

person_3.introduce("안녕하세요.")
person_3.birthday()
person_3.rename("비모")
person_3.introduce("안녕하세요.")

# 인스턴스 / 객체(object) / 클래스
# 인스턴스나 객체의 용어 구분이 애매해졌다.


person_4 = Person("이순신", 20)

Person.species = "인간"

print(person_3.species)
print(person_4.species)

"""
person_1 = "인간"
...
person_100 = "인간"
"""

# 책에 대한 클래스(객체 설계도) 작성
class Book:
    # 생성자 메서드 작성
    def __init__(self, name, page_length):
        # Book 객체에 대한 속성(정보) 정의
        self.name = name  # 책 이름
        self.page_length = page_length  # 책의 페이지 수
        self.page_number = 0  # 현재 페이지 번호

    # 책을 1 페이지 넘기는 기능(메서드) 작성
    def turn(self):
        # 현재 페이지 번호가 책의 페이지 수와 같은지 검사하는 조건문
        if self.page_number != self.page_length:
            # 다르면 페이지를 더 넘길 수 있다.
            # 페이지 번호 + 1한 결과를 페이지 번호 속성에 할당(저장)
            self.page_number = self.page_number + 1

        # 현재 페이지 번호 출력
        print(f"현재 페이지 번호는 {self.page_number}.")

    # 책을 1 페이지 앞으로 넘기는 기능 작성
    def back(self):
        # 유효성 검사(validate)
        if self.page_number != 0:
            self.page_number = self.page_number - 1
        print(f"현재 페이지 번호는 {self.page_number}.")

# Book 객체 생성하고, 변수 book_1에 할당(저장)
# 이름은 파이썬 기본 도서, 페이지 수는 10 인 Book 객체 생성
book_1 = Book("파이썬 기본 도서", 10)
# book_1에 대해서 trun 메서드를 실행
book_1.turn()
book_1.back()

# 책을 넘기는 (앞/뒤) 무한히 넘길 수 있을까요?
# 책의 페이지 수 초과해서 페이지 번호가 증가할 수 있을까?
# 페이지 번호가 0보다 작아질 수 있을까?
for _ in range(100):
    book_1.turn()

for _ in range(100):
    book_1.back()
# 인스타그램은 글 작성 글자 수 제한
# 인스타그램 글 작성 기능에 대한 유효성 검사가 필요하다.
# 사용자가 입력한 글자 수가 제한 글자 수보다 긴지 확인하는 유효성 검사.

# 비밀번호
# 소문자대문자숫자특수문자 10~16자 다 넣어볼까?