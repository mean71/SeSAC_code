# 17_api_2.py
# 학습 날짜 : 9월 2일 금요일
# 파이썬 프리코스 11회차
# 학습 내용 : 공공데이터포탈 오픈API


# Python3 샘플 코드 #


import requests


# 초단기예보 조회 주소
url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst"

# 서비스키를 저장할 변수
# 서비스키는 (Decoding) 서비스키
serviceKey = "서비스키(Decoding)"

# 요청할 때 제공할 추가 정보(파라미터)
parameter = {
    "serviceKey": serviceKey,
    "pageNo": "1",
    "numOfRows": "1000",
    "dataType": "JSon",  # 모든 글자를 대문자로 변환처리하는 코드
    "base_date": "20240902",  # base_date = 20240902
    "base_time": "1900",  #  base_time = 1900
    "nx": "61",
    "ny": "127",
}

response = requests.get(url, params=parameter)

# 딕셔너리(파이썬 자료형)으로 변환한 데이터를 저장
data = response.json()
from pprint import pprint

# 날씨 정보 목록이 저장된 "item" 데이터 추출
item_list = data["response"]["body"]["items"]["item"]

# pprint(item_list)

for item in item_list:
    """
    pprint(item)
    print(type(item))
    """
    # 날씨 중 키 category 정보만 추출
    # 키 category 인덱싱
    category = item["category"]
    # print(category)

    # 찾고 싶은 카테고리 코드 저장 변수
    category_value = "REH"

    # 날씨 데이터 추출 후 출력하는 코드 작성
    # 예측 날짜(fcstDate)
    # 예측 시간(fcstTime)
    # 날씨 정보(fcstValue)
    # 위치 정보(nx / ny)

    # 날씨 종류(category)가 category_value인
    if category == category_value:
        fcstDate = item["fcstDate"]
        fcstTime = item["fcstTime"]
        fcstValue = item["fcstValue"]
        ny = item["ny"]
        nx = item["nx"]
        print(f"예보 날짜 : {fcstDate}")
        print(f"예보 시간 : {fcstTime}")
        print(f"날씨 값 : {fcstValue}")
        print(f"위치 : {ny} / {nx}")
        print("-------------------")


"""
# 응답 데이터의 키 response 에 대한 인덱싱
response_key = data["response"]

# 키 body 에 대한 인덱싱
body_key = response_key["body"]

# 키 items 에 대한 인덱싱
items_key = body_key["items"]

# 키 item 에 대한 인덱싱
item_key = items_key["item"]

pprint(item_key)
"""
