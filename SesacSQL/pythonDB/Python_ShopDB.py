import pymysql
#connect() 함수로 shopDB와 연결자 생성
conn = pymysql.connect(	# 연결자= pymysql.connect(옵션)
  host='localhost',
  user='root',
  password='root',
  db='ShopDB',
  charset='utf8'
  )
'''
    항목			설명
		host			서버IP 주소
		user			사용자
		password	비밀번호
		db				데이터베이스이름
		charset		인코딩
'''
curs = conn.cursor()	# 커서= 연결자.cursor()
'''데이터베이스에SQL문 실행하거나 실행결과를 돌려받는 통로,
앞서 만든 연결자에 cursor()함수 연결하여 생성'''
sql = "SELECT * FROM Product"
curs.execute(sql)         # cur.execute("테이블조회SQL문")SQL문이 데이터베이스에 실행
result = curs.fetchall()	# 모든데이터가져오기
print(type(result))				# result타입확인
for data in result:				#출력
	print(data)
'''
-- [실습] shopDB 데이터조회
		함수							설명
		fetchone() 				cursor 저장데이터 한행씩 추출
		fetchmany(size) 	cursor 저장데이터 size개의 행추출
		fetchall() 				cursor 저장데이터 모두 추출
'''
sql = "SELECT * FROM Product"
curs.execute(sql)

result = curs.fetchmany(2) # 2줄의데이터만가져오기
for data in result: # 출력
    print(data)
# 생략
result = curs.fetchone() # 1줄의데이터만가져오기 fetchone()
print(result) # 출력