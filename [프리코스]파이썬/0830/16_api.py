# 16_api.py
# 학습 날짜 : 8월 30일 금요일
# 파이썬 프리코스 10회차
# 학습 내용 : api

import ctypes

# Windows API를 사용하기 위해 필요한 설정
kernel32 = ctypes.windll.kernel32
hConsole = kernel32.GetStdHandle(-11)


text = "Hello World\n"  # 출력 텍스트
text_buffer = ctypes.create_unicode_buffer(text)
length = len(text)

written = ctypes.c_ulong(0)

# WriteConsoleW API 함수를 호출하여 터미널에 텍스트 출력
kernel32.WriteConsoleW(
    hConsole,
    text_buffer,  # 출력할 텍스트 버퍼
    length,  # 출력할 텍스트 길이
    ctypes.byref(written),
    None,
)

# 파이썬의 print() 함수와 같은 결과 출력
print("Hello World")


# API
# 윈도우 OS, 터미널에 문자를 출력하는 함수(프로그램)
# 위의 함수를 직접적으로 실행할 수는 없다.
# 윈도우 프로그램이 만든 터미널 출력 API를 실행 할 수 있다.

# 시작 메뉴를 여는 함수는 API가 없다고 치면
# 다른 프로그램(파이썬)에서 시작 메뉴를 열 수 있는 방법이 없다.

# 클라이언트 -> 서버에게 이야기 하는것. 요청(request)
# 서버 -> 클라이언트에게 이야기 하는것. 응답(response)
# 이야기 하는것(데이터를 주세요. 데이터를 드릴게요)

# HTTP : 요청과 응답으로 이루어진 규칙(프로토콜)
# 클라이언트와 서버가 데이터를 주고받는 프로토콜인데
# 이때, 요청과 응답을 통해 데이터를 주고받늗다.


# Web API는 HTTP 통신을 통해 실행된다(데이터를 주고받는다)
# HTTP 통신은 클라이언트와 서버의 통신
# 서버를 찾을 때에는 주소(URI,URL)가 필요하다.
# Web API 실행하기 위해서는 서버의 주소가 필요하다.
# (서버의 주소를 통해 Web API를 호출한다.)


# Web API는 무엇인가?
# 데이터를 제공하는 웹 서비스가 데이터를 제공하는 통로(문,길)

# 유튜브
# 영상 데이터
# 조회수
# 좋아요 수
# 제목
# 댓글 수
# 싫어요 수 -> API(통로, 문, 길, 약속)가 없다.

# 데이터를 제외한 Web API는?
# 구글 맵, 네이버 맵, 카카오 맵


# 파이썬의 HTTP 통신을 도와주는 도구
# requests 라이브러리
# 설치 명령어를 터미널에 입력해주세요.
# pip install requests
# 클라이언트 -> 서버, 요청(request)

# requests 라이브러리(모듈) 불러오기
import requests

# 요청(HTTP 통신)을 할 때 필요한것
# 주소(URL, URI)
# 요청 주소 문자열을 생성하고, 변수 url에 저장(할당)
url = "https://jsonplaceholder.typicode.com/posts"

# 웹 브라우저에 주소를 입력하는 동작
# 위 동작에 대한 파이썬 requests 코드
# requests 모듈의 get(주소) 함수를 활용
# 클라이언트 -> 서버 요청을 통해 응답받은 데이터를 변수 res 저장
res = requests.get(url)

# 응답 받은 데이터 출력
# print(res)

# json() 메서드를 실행해서 응답 데이터를 파이썬 데이터로 변환 후
# 변수 data 저장
data = res.json()

# print(data)

#
# 1번 게시글 상세 정보 API 호출
# 주소 : https://jsonplaceholder.typicode.com/posts/1
import requests

url = "https://jsonplaceholder.typicode.com/posts/1"

response = requests.get(url)

data = response.json()
# print(data)

# 잡기술(기능)
# pprint 모듈 (pretty print)
from pprint import pprint

# 출력을 예쁘게 해준다.


# 파이썬 print 결과
"""
{'userId': 1, 'id': 1, 'title': 'sunt aut facere repellat provident occaecati excepturi optio reprehenderit', 'body': 'quia et suscipit\nsuscipit recusandae consequuntur expedita et cum\nreprehenderit molestiae ut ut quas totam\nnostrum rerum est autem sunt rem eveniet architecto'}
"""
# pprint 결과
"""
{'body': 'quia et suscipit\n'
         'suscipit recusandae consequuntur expedita et cum\n'
         'reprehenderit molestiae ut ut quas totam\n'
         'nostrum rerum est autem sunt rem eveniet architecto',
 'id': 1,
 'title': 'sunt aut facere repellat provident occaecati excepturi optio '
          'reprehenderit',
 'userId': 1}
"""
# 데이터 구조가 복잡한 데이터를 출력하기 좋다.
# 데이터 구조가 복잡하다 -> 리스트안의 리스트, 리스트안의 딕셔너리, 딕셔너리안의 딕셔너리


import requests

url = "https://jsonplaceholder.typicode.com/posts/1"

response = requests.get(url)

text_data = response.text
json_data = response.json()
print(type(text_data))  # 문자열 str
print(type(json_data))  # 딕셔너리 dict

pprint(json_data)
# 게시글의 타이틀(title) 출력
pprint(json_data["title"])

# pprint(text_data["title"])
