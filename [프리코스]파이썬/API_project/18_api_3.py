# 18_api_3.py
# 학습 날짜 : 9월 3일 화요일
# 파이썬 프리코스 12회차
# 학습 내용 : TMDB API

# API Key와 Token은 무엇이 다른가?
# 1. API Key 계정에 종속 X
# 2. Token 계정에 종속 O (계정을 대표)
# 예시, API Key와 Token 재생성하는 경우
# (구) API Key와 (신) API Key 다른 Key
# (구) Token / (신) Token 하나의 계정을 대표
# 즐겨찾기, 좋아요 -> 계정의 정보


headers = {
    # 내가 데이터를 받고 싶은데 ~~~한 데이터 형태를 받고 싶어
    # XML / JSON
    "accept": "application/json",
    # 인증 : 인증 토큰(액세스 토큰)
    # Bearer 인증 방식을 의미
    # 02 집 전화기, 010 휴대폰, 070 인터넷 전화
    # 도로명 주소, 지번 주소
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI3YmY4ZjNlNDVmZDIzNWZkOTljMjA4YjYxZGNmZGVjYSIsIm5iZiI6MTcyNTM1ODcwMy43MTk0MjQsInN1YiI6IjVkODMyOWJmMTYyYmMzMDIyN2RkZmNkNiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Wh-NKR3DDlBHlAw2BYrJrqRbatV6fhHme9ookeS0BsM",
}

# API 주소을 만들 때 버전을 명시한다.
# 버전1 -> 버전2 할 때, 버전을 명시안하면
# 버전1에 대한 API가 버전2로 대체된다.
# 버전1에 대한 주소는 더이상 사용할 수 없다.

# TMDB API 버전 1을 사용하던 다른 프로그램, 웹 서비스 먹통 가능성
# 버전1에서는 1/popular/movie
# 버전2에서는 2/movie/popular


import requests

# 주소(url) 분리
# domain과 file path, parameter 분리
domain = "https://api.themoviedb.org"

# 찾고 싶은 영화 번호
movie_id = 2000

# 찾고 싶은 배우 번호
person_id = 100

# API 엔드포인트(API 기능 주소)
path = f"/3/movie/now_playing"

# 파라미터, 추가 입력 정보
params = {
    "language": "ko-KR",
    "page": "1",
}

# 주소 조합
url = domain + path

headers = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI3YmY4ZjNlNDVmZDIzNWZkOTljMjA4YjYxZGNmZGVjYSIsIm5iZiI6MTcyNTM1ODcwMy43MTk0MjQsInN1YiI6IjVkODMyOWJmMTYyYmMzMDIyN2RkZmNkNiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Wh-NKR3DDlBHlAw2BYrJrqRbatV6fhHme9ookeS0BsM",
}

response = requests.get(url, headers=headers, params=params)

# json() 메서드를 활용해서 파이썬 자료형(딕셔너리 / 리스트)으로 변환
data = response.json()


import requests

TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI3YmY4ZjNlNDVmZDIzNWZkOTljMjA4YjYxZGNmZGVjYSIsIm5iZiI6MTcyNTM1ODcwMy43MTk0MjQsInN1YiI6IjVkODMyOWJmMTYyYmMzMDIyN2RkZmNkNiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Wh-NKR3DDlBHlAw2BYrJrqRbatV6fhHme9ookeS0BsM"

# 주소(url) 분리
# url = "https://api.themoviedb.org/3/movie/now_playing?language=en-US&page=1"

# domain / file path / parameter 분리
domain = "https://api.themoviedb.org"

# API 엔드포인트(API 기능 주소)
path = f"/3/movie/now_playing"

# 파라미터, 추가 입력 정보
params = {
    "language": "ko-KR",
    "page": "1",
}

# 주소 조합
url = domain + path

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {TOKEN}",
}

# API 호출
response = requests.get(url, headers=headers, params=params)

# json() 메서드를 활용해서 파이썬 자료형(딕셔너리 / 리스트)으로 변환
data = response.json()

from pprint import pprint


# 영화 정보 목록(results)만 추출
results = data["results"]
# pprint(results)

# 영화 정보 목록의 길이
results_len = len(results)
# print(results_len)

# 첫 3개를 슬라이싱
three_results = results[0:3]
# pprint(three_results)

# 첫 3개의 영화에 대한 영화 제목(title) 추출
for movie in three_results:
    title = movie["title"]
    overview = movie["overview"]
    vote_average = movie["vote_average"]

    print(f"영화 제목 : {title}")
    print(f"평균 평점 : {vote_average}")
    print(f"영화 줄거리")
    print(overview)
    print("------------------------")


# 영화 중 평균 평점이 가장 높은 영화 찾기

# 평균 평점 기준값을 활용한 가장 큰 평균 평점 탐색
# 반복문을 수행하면서 가장 큰 평균 평점을 저장할 변수
max_vote = 0

for movie in results:
    # 영화 정보 중 평균 평점만 추출
    vote_average = movie["vote_average"]
    print(vote_average)

    # max_vote와 각 영화의 평균 평점(vote_average)을 크기 비교
    # 새로운 가장 큰 평균 평점이 나타난다면
    # 기준값(max_vote)의 값을 갱신
    if max_vote < vote_average:
        max_vote = vote_average

print("------")
print(max_vote)


# 가장 큰 평균 평점(max_vote)와 동일한 영화 탐색
for movie in results:
    vote_average = movie["vote_average"]
    # 영화의 평균 평점이
    # 가장 큰 평균 평점과 같다면
    # 영화 정보를 출력
    if max_vote == vote_average:
        print(movie)


# 평균 평점 기준값을 활용한 가장 작은 평균 평점 탐색
# 반복문을 수행하면서 가장 작은 평균 평점을 저장할 변수
min_vote = 10

# 가장 작은 평점의 영화를 저장할 변수
min_movie = {}

for movie in results:
    # 영화 정보 중 평균 평점만 추출
    vote_average = movie["vote_average"]
    print(vote_average)

    # min_vote와 각 영화의 평균 평점(vote_average)을 크기 비교
    # 새로운 가장 작은 평균 평점이 나타난다면
    # 기준값(min_vote)의 값을 갱신
    # 가장 작은 평점의 영화(min_movie)의 값을 갱신
    if min_vote > vote_average:
        min_vote = vote_average
        min_movie = movie

pprint(min_movie)
