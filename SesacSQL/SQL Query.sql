CREATE DATABASE TestDB;
USE TestDB;
DROP DATABASE TestDB;
# DB생성>사용>삭제
CREATE DATABASE BookDB;
USE BookDB;
CREATE TABLE TestTable(
	col_1 INT,
    col_2 VARCHAR(20),
    col_3 DATETIME
);
DROP TABLE TestTable;
# TestTable 생성,삭제
CREATE TABLE Book(
	bookid		INT,
	bookname	VARCHAR(20),
    publisher	VARCHAR(20),
    price		INT
);# book Table 생성
INSERT INTO Book (bookid, bookname, publisher, price)
VALUES
(100,'데이터베이스','한빛아카데미',27000),
(101, '파이썬', '한빛아카데미', 22000),
(102, 'JSP 프로그래밍', '생능출판사', 26000),
(103, '자바스크립트', '길벗', 45000),
(104, '데이터베이스 배움터', '생능출판사', 30000);
SELECT * FROM Book;
SELECT bookname FROM Book;#'Book'테이블에 저장된 bookname조회
SELECT publisher FROM Book;#'Book'테이블에 저장된 publisher조회
SELECT bookname, publisher FROM Book;#'Book' 테이블에 저장된 bookname, publisher를 조회
#조건으로 데이터 검색
SELECT * FROM Book
WHERE price > 25000;#가격이 25,000원 이상인 책 정보 검색
SELECT * FROM Book
WHERE price BETWEEN 10000 AND 25000;#가격이 10,000원 이상 25,000원 이하인 책 정보 검색
SELECT * FROM Book
WHERE publisher IN('길벗', '생능출판사');#출판사가 '길벗' 또는 '생능출판사'인 책 정보 검색
SELECT * FROM Book
WHERE publisher NOT IN('길벗', '생능출판사');#출판사가 '길벗' 또는 '생능출판사'가 아닌 책 정보 검색
SELECT publisher FROM Book
WHERE bookname LIKE '데이터베이스';#'데이터베이스'를 출간한 출판사 검색
SELECT bookname, publisher FROM Book
WHERE bookname LIKE '%데이터베이스%';# 책 이름에 '데이터베이스'가 포함된 책 이름과 출판사 검색
SELECT bookname FROM Book
WHERE bookname LIKE '_이%';#왼쪽 두 번째 위치에 '이'라는 문자를 갖는 책 이름을 검색
SELECT * FROM Book
WHERE publisher='길벗' OR publisher='생능출판사';#출판사가 '길벗' 또는 '생능출판사'인 책 정보 검색
SELECT * FROM Book
WHERE bookname LIKE '%데이터%' AND price >= 30000;#데이터 관련 도서 중, 가격이 30,000원 이상인 책 검색
#데이터정렬
SELECT * FROM Book ORDER BY bookname ASC;#[실습] ORDER BY-1 도서를 이름순으로 검색(오름차순)
SELECT * FROM Book ORDER BY bookname;#[실습] ORDER BY-2 도서를 이름순으로 검색(오름차순)
SELECT * FROM Book ORDER BY price;#[실습] ORDER BY-3 도서를 가격순으로 검색(오름차순)
SELECT * FROM Book ORDER BY bookname DESC;#[실습] ORDER BY-4 도서를 이름순으로 검색(내림차순)
SELECT * FROM Book ORDER BY price DESC;#[실습] ORDER BY-4 도서를 가격순으로 검색(내림차순)
#집계함수
SELECT SUM(price) FROM Book;# [실습] SUM 함수 : 전체 도서 가격의 합 출력
SELECT SUM(price) AS '가격 총합' FROM Book;# [실습] SUM 함수 : 전체 도서 가격의 합 출력 - 열 제목 추가
SELECT AVG(price) AS '평균 가격' FROM Book;# [실습] AVG 함수 : 전체 도서들의 평균 가격 출력
SELECT COUNT(*) AS '총 개수' FROM Book;# [실습] COUNT 함수 : 도서의 총 개수 출력
SELECT MIN(price) AS '최저가' FROM Book;# [실습] MIN 함수 : 최저가 도서 계산
SELECT MAX(price) AS '최고가' FROM Book;# [실습] MAX 함수 : 최고가 도서 계산
SELECT bookname AS '책제목' FROM book
WHERE price = (SELECT MIN(price) FROM book);###########
#데이터 그룹화
INSERT INTO Book (bookid, bookname, publisher, price)
VALUES
(105,'HTML기초','한빛아카데미',37000),
(106,'파이썬 데이터분석','이지스퍼블리싱',25000),
(107,'Chat GPT','생능출판사',29000),
(108,'ReactJS','이지스퍼블리싱',41000),
(109,'홈페이지 만들기 기초','한빛아카데미',32000),
(110,'데이터 시각화','생능출판사',27000);#Book테이블에 데이터 추가

SELECT publisher, sum(price) FROM Book
GROUP BY publisher;# [실습] GROUP BY : 각 출판사별 도서 금액의 총합 계산 #주의사항 : GROUP BY에서 사용한 속성과 집계함수만 올 수 있음
#SELECT bookname, sum(price) FROM Book
#GROUP BY publisher;# [실습] GROUP BY : 각 출판사별 도서 금액의 총합 계산 #에러발생 이유 : bookname 속성을 사용하면 안됨
SELECT publisher, sum(price) FROM Book
GROUP BY publisher
HAVING publisher = '생능출판사';# [실습] HAVING : 출판사를 기준으로 그룹화 후,'생능출판사'의 데이터만 검색하여 도서 금액의 총합 계산


#SELECT publisher, sum(price) FROM Book
#GROUP BY publisher		#에러발생 이유 : 그룹화에 사용하지 않은 열을 사용하면 안됨 아래로 수정
#HAVING price > 20000; # [실습] HAVING : 출판사를 기준으로 그룹화 후, 가격이 20,000원 이상인 도서만 검색하여 도서 금액의 총합 계산
SELECT publisher, sum(price) FROM Book
WHERE price > 20000
GROUP BY publisher;# [실습] HAVING : 출판사를 기준으로 그룹화 후, 가격이 20,000원 이상인 도서만 검색하여 도서 금액의 총합 계산#에러수정
# 중복데이터제거
SELECT DISTINCT publisher FROM Book; #[실습] DISTINCT : 출판사 이름의 중복을 제거하여 검색
# 데이터수정
SELECT * FROM book;# [실습] UPDATE : book 테이블 전체 데이터 조회
UPDATE Book SET price = 23000;# [실습] UPDATE : 책의 가격을 23,000원으로 수정
SELECT * FROM book;
UPDATE Book SET price = 23000
WHERE bookid = 101;# [실습] UPDATE : bookid가 101인 책의 가격을 23,000원으로 수정
SELECT * FROM book;
# 데이터삭제
DELETE FROM Book WHERE publisher = '길벗';# [실습] DELETE : '길벗' 출판사의 데이터 삭제
SELECT * FROM book;
# 테이블속성변경
ALTER TABLE Book ADD isbn VARCHAR(10);# [실습] ADD : Book 테이블에 다음 속성 추가 : VARCHAR(10) 자료형을 가진 isbn 속성 추가
ALTER TABLE Book MODIFY isbn INT;# [실습] MODIFY : Book 테이블의 isbn 속성의 데이터 타입을 INT로 변경
ALTER TABLE Book CHANGE isbn 일련번호 INT;# [실습] CHANGE : Book 테이블의 isbn 속성의 이름을 '일련번호'로 변경
ALTER TABLE Book DROP COLUMN 일련번호;# [실습] DROP : Book 테이블의 일련번호 속성을 삭제

#기본키
# [실습] 기본키-1 : newBook 테이블 생성하기
CREATE TABLE newBook(
	    bookid INT PRIMARY KEY,
	    bookname VARCHAR(20),
	    publisher VARCHAR(20),
	    price INT
	);
# [실습] 기본키-2 : 데이터 삽입하기
INSERT INTO newBook (bookid, bookname, publisher, price)
VALUES (100, '데이터베이스', '한빛아카데미', 27000);
# [실습] 기본키-3 : 데이터 삽입하기
INSERT INTO newBook (bookid, bookname, publisher, price)
VALUES (100, '프로그래밍', '한빛아카데미', 30000);
# [실습] 기본키-4 : 데이터 삽입하기
INSERT INTO newBook (bookid, bookname, publisher, price)
VALUES (NULL, '데이터 시각화', '생능출판사', 27000);

# 데이터 삽입하기
INSERT INTO newBook (bookid, bookname, publisher, price)
VALUES (101, NULL, NULL, 25000);
# 기타 제약조건 - NULL, NOT NULL
# [실습] NOT NULL :  테이블 속성 변경하기 : bookname, publisher 속성에 NOT NULL 제약 조건 추가
ALTER TABLE newBook MODIFY bookname VARCHAR(20) NOT NULL;
ALTER TABLE newBook MODIFY publisher VARCHAR(20) NOT NULL;
# [실습] NOT NULL : 데이터 삽입하기
INSERT INTO newBook (bookid, bookname, publisher, price)
VALUES (102, NULL, NULL, 25000);
# [실습] NOT NULL : 데이터 삽입하기
	INSERT INTO newBook (bookid, bookname, publisher, price)
	VALUES (102, '데이터 시각화', '생능출판사', 25000);
# DEFAULT
# [실습] DEFAULT : 테이블 속성 변경하기 : price 속성의 기본값을 10,000원으로 지정
ALTER TABLE newBook MODIFY price INT DEFAULT 10000;
# [실습] DEFAULT :  데이터 삽입하기
INSERT INTO newBook (bookid, bookname, publisher)
VALUES (103, '프로그래밍', '한빛아카데미');

# 외래키
# newBook 테이블에 데이터 입력하기 : 기존 데이터를 모두 제거 후, 입력
DELETE FROM newBook;
INSERT INTO newBook (bookid, bookname, publisher, price) VALUES (100, '데이터베이스', '한빛아카데미', 27000);
INSERT INTO newBook (bookid, bookname, publisher, price) VALUES (101, '파이썬', '한빛아카데미', 22000);
INSERT INTO newBook (bookid, bookname, publisher, price) VALUES (102, 'JSP 프로그래밍', '생능출판사', 26000);
INSERT INTO newBook (bookid, bookname, publisher, price) VALUES (103, '자바스크립트', '길벗', 45000);
INSERT INTO newBook (bookid, bookname, publisher, price) VALUES (104, '데이터베이스 배움터', '생능출판사', 30000);
INSERT INTO newBook (bookid, bookname, publisher, price) VALUES (105, 'HTML 기초', '한빛아카데미', 37000);
INSERT INTO newBook (bookid, bookname, publisher, price) VALUES (106, '파이썬 데이터', '이지스퍼블리싱', 25000);
INSERT INTO newBook (bookid, bookname, publisher, price) VALUES (107, 'Chat GPT', '생능출판사', 29000);
INSERT INTO newBook (bookid, bookname, publisher, price) VALUES (108, 'ReactJS', '이지스퍼블리싱', 41000);
INSERT INTO newBook (bookid, bookname, publisher, price) VALUES (109, '홈페이지 만들기', '한빛아카데미', 32000);
INSERT INTO newBook (bookid, bookname, publisher, price) VALUES (110, '데이터 시각화', '생능출판사', 27000);
# [실습] 외래키-1 : newOrders 테이블 생성하기 : bookid 속성을 외래키로 지정
CREATE TABLE newOrders(
orderid VARCHAR(10) PRIMARY KEY,
bookid INT NOT NULL,
member VARCHAR(10) NOT NULL,
address VARCHAR(20) NOT NULL,
FOREIGN KEY(bookid) REFERENCES newBook(bookid)
);
# [실습] 외래키-2 : 데이터 삽입하기
INSERT INTO newOrders(orderid, bookid, member, address)
VALUES ('p001', 102, '정수아', '서울');
# [실습] 외래키-3 : 데이터 삽입하기 # 에러 이유 : 외래키에 입력되는 값은 참조테이블에 입력된 값만 가능
INSERT INTO newOrders(orderid, bookid, member, address)
VALUES ('p002', 120, '정수아', '서울');	
# [실습] 외래키-4 : 참조 테이블의 데이터 삭제하기 : 에러 이유 : newOrders 테이블에서 bookid 속성을 참조하기 때문
DELETE FROM newBook WHERE bookid=102;

# newOrders 테이블에 데이터 입력하기 : 기존 데이터를 모두 제거 후, 입력
DELETE FROM newOrders;
INSERT INTO newOrders(orderid, bookid, member, address) VALUES ('p001', 102, '오한솔', '경기');
INSERT INTO newOrders(orderid, bookid, member, address) VALUES ('p002', 107, '김현우', '서울');
INSERT INTO newOrders(orderid, bookid, member, address) VALUES ('p003', 103, '박홍진', '부산');
INSERT INTO newOrders(orderid, bookid, member, address) VALUES ('p004', 102, '김현우', '서울');
INSERT INTO newOrders(orderid, bookid, member, address) VALUES ('p005', 104, '문종헌', '대전');
INSERT INTO newOrders(orderid, bookid, member, address) VALUES ('p006', 105, '김현우', '서울');
INSERT INTO newOrders(orderid, bookid, member, address) VALUES ('p007', 103, '이봉림', '부산');
INSERT INTO newOrders(orderid, bookid, member, address) VALUES ('p008', 102, '정희성', '경기');
INSERT INTO newOrders(orderid, bookid, member, address) VALUES ('p009', 107, '오한솔', '경기');
INSERT INTO newOrders(orderid, bookid, member, address) VALUES ('p010', 103, '김현우', '서울');
#--- 2개의 테이블을 이용한 조인 검색 : 김현우 고객이 주문한 책의 제목과 가격 검색
SELECT A.a, B.d FROM A, B
WHERE 조건 AND A.b = B.b;
# 김현우 고객이 주문한 책의 제목과 가격 검색
SELECT newBook.bookname, newBook.price
FROM newBook, newOrders
WHERE newOrders.member = '김현우'
AND newOrders.bookid = newBook.bookid; 
# [실습] 2개의 테이블을 이용한 조인 검색 : 도서를 주문한 고객들의 운송장 정보 출력
SELECT neworders.orderid, neworders.member,
newBook.bookname, newBook.price, neworders.address
FROM newBook, newOrders
WHERE newOrders.bookid = newBook.bookid;

# 1-1 학생 데이터베이스(StudentDB) 생성
CREATE DATABASE StudentDB;
USE StudentDB;
#1-2 학생 정보 테이블(StudentInfo) 생성
CREATE TABLE StudentInfo(
	id int,
	name varchar(20),
    age int,
    address varchar(20),
    course varchar(20)
);
#1-3 학생 정보 입력
INSERT INTO StudentInfo (id, name, age, address, course)
VALUES
(1, '문종헌', 24, '서울', '영어'),
(2, '오한솔', 22, '부산', '수학'),
(3, '정국철', 25, '서울', '음악'),
(4, '박기석', 27, '대전', '국어'),
(5, '안창범', 20, '광주', '수학'),
(6, '박홍진', 22, '부산', '컴퓨터'),
(7, '공지훈', 28, '강원', '국어'),
(8, '정희성', 30, '제주', '음악'),
(9, '이봉림', 34, '대전', '영어'),
(10, '김현우', 21, '서울', '컴퓨터');
SELECT * FROM StudentInfo;
#1-4 전체 학생의 id와 이름을 검색
SELECT id, name FROM StudentInfo;

# 3-1 나이가 30 이상인 학생 정보 검색
SELECT *from StudentInfo WHERE age > 30;
#3-2 '컴퓨터'를 수강하는 학생 정보 검색
SELECT * FROM StudentInfo WHERE course = '컴퓨터';
#3-3이름이 '박기석'인 학생 정보 검색
SELECT * FROM StudentInfo WHERE name = '박기석';
#3-4나이가 20~25살 사이인 학생 정보 검색 BETWEEN
SELECT * FROM StudentInfo WHERE age BETWEEN 20 AND 25;
#3-5 나이가 28살이거나 34살인 학생 정보 검색 IN
SELECT * FROM StudentInfo WHERE age IN(28, 34);
#3-6성이 '김'씨인 학생 정보 검색
SELECT * FROM StudentInfo WHERE name LIKE '김%';
#3-7이름의 두 번째 글자가 '지'이고, 그 뒤는 무엇이든 관계없는 학생 정보 검색
SELECT * FROM StudentInfo WHERE name LIKE '_지%';
#3-8나이를 기준으로 오름차순 정렬하여 검색
SELECT * FROM StudentInfo ORDER BY age;
#3-9나이가 많은 사람부터 적은 사람 순으로 순차적으로 검색
SELECT * FROM StudentInfo ORDER BY age DESC;

# 5-1 '박기석' 학생의 주소를 '제주'로 변경
UPDATE StudentInfo SET address = '제주'
WHERE name = '박기석';
# 5-2  id가 10인 학생을 삭제
DELETE FROM StudentInfo 
WHERE ID = 10;
# 5-3 학생 정보 테이블에 새로운 속성 추가 : score(문자형 최대 2자)
ALTER TABLE StudentInfo
ADD score VARCHAR(2);
# 5-4 각 학생의 score 속성에 학점 값 삽입 : 순서대로 A, B, A, C, B, D, A, C, D, A 입력
UPDATE StudentInfo SET score = 'A' WHERE id = 1;
UPDATE StudentInfo SET score = 'B' WHERE id = 2;
UPDATE StudentInfo SET score = 'A' WHERE id = 3;
UPDATE StudentInfo SET score = 'C' WHERE id = 4;
UPDATE StudentInfo SET score = 'B' WHERE id = 5;
UPDATE StudentInfo SET score = 'D' WHERE id = 6;
UPDATE StudentInfo SET score = 'A' WHERE id = 7;
UPDATE StudentInfo SET score = 'C' WHERE id = 8;
UPDATE StudentInfo SET score = 'D' WHERE id = 9;
UPDATE StudentInfo SET score = 'A' WHERE id = 10;
# 5-5 각 학점 별 학생 수 계산
SELECT score, COUNT(*) AS student_count
FROM StudentInfo
GROUP BY score;
# 5-6  '컴퓨터' 또는 '영어' 과목을 수강하는 학생의 이름 및 과목명
SELECT name, course FROM StudentInfo
WHERE course IN('컴퓨터','영어');

#2-1 제목이 '프로그래밍'으로 끝나는 책 정보 검색 : 제목의 길이는 상관 없음 : 책 제목, 가격만 조회
SELECT bookname, price FROM Book
WHERE bookname LIKE '%프로그래밍';
#2-2 제목이 '데이터'로 시작하면서 6자인 책 정보 검색 : 책의 모든 정보 조회
SELECT * FROM Book
WHERE bookname LIKE '데이터___';
#2-3 제목의 세 번째 글자가 '터'인 책 정보 검색
SELECT * FROM Book
WHERE bookname LIKE '__터%';
#2-4 '한빛아카데미'의 도서 중 '데이터' 관련된 책 정보 검색
SELECT * FROM Book
WHERE bookname LIKE '%데이터%' AND publisher='한빛아카데미';

# 4-1 데이터베이스 관련 책 가격의 총액 계산
SELECT sum(price) AS '총액' from Book
WHERE bookname like'%데이터베이스%';
# 4-2 출판사가 '한빛아카데미'인 책의 개수 출력
SELECT COUNT(publesher)AS '한빛아카데미' FROM Book
WHERE publisher in('한빛아카데미');
# 4-3 출판사 별로 가격이 30,000원 이상인 도서의 총 수량을 계산
SELECT publisher, COUNT(*) AS 도서수량 FROM Book
WHERE price >= 30000
GROUP BY publisher;
# 4-4 출판사 별로 가격이 30,000원 이상인 도서의 총 수량을 계산:총 수량이 두 권 이상인 출판사만 조회
SELECT publisher, count(*)AS 도서수량 from Book
WHERE price >= 30000
GROUP BY publisher;
# 4-5 학생들의 나이 총 합 계산
SELECT SUM(age) AS '나이총합' FROM StudentInfo;
# 4-6 중복을 제거한 과목명 검색
SELECT DISTINCT course FROM StudentInfo;
# 4-7 '컴퓨터'를 수강하는 학생들의 평균 나이 계산
SELECT AVG(age) AS '평균나이' FROM StudentInfo
WHERE course='컴퓨터';
# 4-8 '영어' 과목을 수강하는 학생 수를 계산
SELECT COUNT(*) AS '수강인원' FROM StudentInfo
WHERE course='영어';
# 4-9 각 지역별 학생 수를 계산
SELECT address, COUNT(*) AS '학생수'
FROM StudentInfo
GROUP BY address;
# 4-10 각 지역별 학생들의 평균 나이를 계산
SELECT address, AVG(age) AS '평균나이'
FROM StudentInfo
GROUP BY address;
# 4-11 과목 별 평균 나이가 25세 이상인 과목명과 학생 수를 계산:HAVING 이용
SELECT course, COUNT(*) AS '학생수' FROM StudentInfo
GROUP BY course
HAVING AVG(age) >= 25;
#68P 입력추가
#6-1 도서 제목에 '데이터'가 포함된 도서를 'Data'로 변경한 후 도서 목록을 출력
UPDATE Book SET bookname = 'Data'
WHERE bookname LIKE '%Data%';
SELECT * FROM Book;
#6-2 한빛아카데미에서 출판한 도서의 제목과 제목의 문자 수, 바이트 수를 출력
SELECT bookname, CHAR_LENGTH(bookname) AS 문자수, LENGTH(bookname) AS 바이트수
FROM Book
WHERE publisher = '한빛아카데미';
#6-3 Book 테이블에 새로운 속성 추가 : 주문 날짜 : orderdate(DATE 타입)
ALTER TABLE Book ADD orderdate DATE;
#6-4 각 도서에 주문 날짜 데이터 추가 후, 목록을 출력
UPDATE Book SET orderdate = '2024-01-01' WHERE bookid = 100;
UPDATE Book SET orderdate = '2024-01-02' WHERE bookid = 101;
UPDATE Book SET orderdate = '2024-01-03' WHERE bookid = 102;
UPDATE Book SET orderdate = '2024-01-04' WHERE bookid = 103;
UPDATE Book SET orderdate = '2024-01-05' WHERE bookid = 104;
UPDATE Book SET orderdate = '2024-01-06' WHERE bookid = 105;
UPDATE Book SET orderdate = '2024-01-07' WHERE bookid = 106;
UPDATE Book SET orderdate = '2024-01-08' WHERE bookid = 107;
UPDATE Book SET orderdate = '2024-01-09' WHERE bookid = 108;
UPDATE Book SET orderdate = '2024-01-10' WHERE bookid = 109;
UPDATE Book SET orderdate = '2024-01-11' WHERE bookid = 110;
SELECT * FROM Book;
#6-5 주문일로부터 10일 후 매출을 확정한다고 할 때, 각 주문의 확정 일자를 계산
SELECT bookname, orderdate, DATE_ADD(orderdate, INTERVAL 10 DAY) AS 주문확정일자
FROM Book;





# 7번째(실습-관광데이터)
# 한국 관광 데이터랩 데이터 가져오기 : 이동통신, 신용카드, 네비게이션 등 다양한 관광 빅데이터 및 융합 분석 서비스를 제공하는 관광특화 빅데이터 플랫폼
# 전국 관광지 검색 순위 데이터 가져오기
# 관광 데이터베이스(tourDB) 생성,사용
CREATE DATABASE tourDB;
USE tourDB;
SELECT * FROM tourTable;
# CSV 데이터 가져오기 : PDF excell
# 테이블 생성하기
# 데이터 입력하기
# 데이터 조회하기 : tourTable 테이블 데이터 조회하기
SELECT * FROM tourTable;
# 문제1 : ❖ 시도명을 한번만 출력 • 2가지 방식으로 작성할 것
#SELECT DISTINCT 시도명 FROM tourTable;
SELECT 시도명 FROM tourTable
GROUP BY 시도명;
# 문제2 : ❖ 경기도의 관광지 정보 검색 • 2가지 방식으로 작성할 것
#SELECT * FROM tourTable
#WHERE 시도명 = '경기도';
SELECT * FROM tourTable
WHERE 시도명 LIKE '%경기도%';
# 문제3,4 : ❖ 쇼핑 분야의 관광지 수 계산❖ 각 분야 별 관광지 수 계산
SELECT COUNT(*) AS '쇼핑 관광지 수' FROM tourTable
WHERE 중분류 = '쇼핑';
#4
SELECT 중분류, COUNT(*) AS '관광지 수' FROM tourTable
GROUP BY 중분류;
# 문제5,6 : ❖ 테마공원의 이름 및 주소 검색❖ 검색건수가 60만 건 이상인 관광지 수 계산
SELECT 관광지명, 주소 FROM tourTable
WHERE 소분류 = '테마공원';
# 6 : 
SELECT 소분류, COUNT(*) AS '관광지 수' FROM tourTable
WHERE 검색건수 > 600000
GROUP BY 소분류;
# 문제7 : ❖ 가장 인기가 없는 관광지부터 인기가 많은 순으로 조회
SELECT * FROM tourTable
ORDER BY 검색건수 ASC;
# 문제8 : ❖ 10개의 관광지 정보만 조회 • LIMIT를 이용할 것
SELECT * FROM tourTable
LIMIT 10;

CREATE DATABASE tourDB;
USE tourDB;-- tourTable 테이블 데이터 조회하기




-- 경기도의 관광지 정보 검색1
SELECT * FROM tourTable
WHERE 시도명 = '경기도';
-- 경기도의 관광지 정보 검색2
SELECT * FROM tourTable
WHERE 주소 LIKE '%경기%';
-- 쇼핑 분야의 관광지 수 계산
SELECT COUNT(*) AS '쇼핑' FROM tourTable
WHERE 중분류 = '쇼핑';
-- 각 분야 별 관광지 수 계산
SELECT 중분류, COUNT(*) AS '관광지 수' FROM tourTable
GROUP BY 중분류;
-- 테마공원의 이름 및 주소 검색
SELECT 관광지명, 주소 FROM tourTable
WHERE 중분류 = '문화관광' AND 소분류 = '테마공원';
-- 검색건수가 60만 건 이상인 관광지 수 계산
SELECT 소분류, COUNT(*) AS '관광지 수' FROM tourTable
WHERE 검색건수 > 600000
GROUP BY 소분류;


#LIMIT n:
# • 0부터 시작하여 n개까지의 데이터 검색
# • LIMIT 0, n과 같은 의미
# LIMIT n, m • n부터 시작하여 m개의 데이터를 검색
# limit 4,10(4번째레코드부터 10개의 레코드가져오기)
	
# 실습 판매DB
#1. 판매DB 및 테이블 생성
CREATE TABLE 고객(
	아이디	VARCHAR(20)		PRIMARY KEY,
    이름		VARCHAR(10)		NOT NULL,
    나이		INT,
    등급		VARCHAR(10)		NOT NULL,
    직업		VARCHAR(20),
    적립금	INT				DEFAULT 0
);
CREATE TABLE 제품(
	제품번호	CHAR(3)			PRIMARY KEY,
	제품명	VARCHAR(20),
    제고량	INT,
    단가		INT,
    제조업체	VARCHAR(20)
);
CREATE TABLE 주문 (
	주문번호	CHAR(3)			PRIMARY KEY,
    주문고개	VARCHAR(20),
    주문제품	CHAR(3),
    수량		INT,
    배소지	VARCHAR(30),
    주문일자	DATE,
    FOREIGN KEY(주문고객) REFERENCES 고객(아이디),
    FOREIGN KEY(주문제품) REFERENCES 제품(제품번호)
);
#2. 각 테이블에 데이터 입력
INSERT INTO 고객 VALUES ('apple', '정소화', 20, 'gold', '학생', 1000);
INSERT INTO 고객 VALUES ('banana', '김선우', 25, 'vip', '간호사', 2500);
INSERT INTO 고객 VALUES ('carrot', '고명석', 28, 'gold', '교사', 4500);
INSERT INTO 고객 VALUES ('orange', '김용욱', 22, 'silver', '학생', 0);
INSERT INTO 고객 VALUES ('melon', '성원용', 35, 'gold', '회사원', 5000);
INSERT INTO 고객 VALUES ('peach', '오형준', NULL, 'silver', '의사', 300);    
INSERT INTO 고객 VALUES ('pear', '채광주', 31, 'silver', '회사원', 500);

INSERT INTO 제품 VALUES('p01', '그냥만두', 5000, 4500, '대한식품');
INSERT INTO 제품 VALUES('p02', '매운쫄면', 2500, 5500, '민국푸드');
INSERT INTO 제품 VALUES('p03', '쿵떡파이', 3600, 2600, '한빛제과');
INSERT INTO 제품 VALUES('p04', '맛난초콜릿', 1250, 2500, '한빛제과');
INSERT INTO 제품 VALUES('p05', '얼큰라면', 2200, 1200, '대한식품');
INSERT INTO 제품 VALUES('p06', '통통우동', 1000, 1550, '민국푸드');
INSERT INTO 제품 VALUES('p07', '달콤비스킷', 1650, 1500, '한빛제과');

INSERT INTO 주문 VALUES('o01', 'apple', 'p03', 10, '서울시 마포구', '2022-01-01');
INSERT INTO 주문 VALUES('o02', 'melon', 'p01', 5, '인천시 계양구', '2022-01-10');
INSERT INTO 주문 VALUES('o03', 'banana', 'p06', 45, '경기도 부천시', '2022-01-11');
INSERT INTO 주문 VALUES('o04', 'carrot', 'p02', 8, '부산시 금정구', '2022-02-01');
INSERT INTO 주문 VALUES('o05', 'melon', 'p06', 36, '경기도 용인시', '2022-02-20');
INSERT INTO 주문 VALUES('o06', 'banana', 'p01', 19, '충청북도 보은군 마포구', '2022-03-02');
INSERT INTO 주문 VALUES('o07', 'apple', 'p03', 22, '서울시 영등포구', '2022-03-15');
INSERT INTO 주문 VALUES('o08', 'pear', 'p02', 50, '강원도 춘천시', '2022-04-10');
INSERT INTO 주문 VALUES('o09', 'banana', 'p04', 15, '전라남도 목포시', '2022-04-11');
INSERT INTO 주문 VALUES('o10', 'carrot', 'p03', 20, '경기도 안양시', '2022-05-22');

#3. 고객 테이블에서 검색
#3-1. 고객 테이블에서 아이디, 이름, 등급 검색하세요.
SELECT 아이디, 이름, 등급 FROM 고객;
#3-2. 고객 테이블에서 성이 김씨인 고객의 이름, 나이, 등급, 적립금을 검색하세요.
SELECT 이름, 나이, 등급, 적립금 FROM 고객 WHERE 이름 LIKE '김%';
#3-3. 고객 테이블에서 아이디가 5자인 고객의 아이디, 이름, 등급을 검색하세요.
SELECT 아이디, 이름, 등급 FROM 고객 WHERE 아이디 LIKE '_____';
#3-4. 고객 테이블에서 나이가 아직 입력되지 않은 고객의 이름을 검색하세요. - IS 키워드 이용
SELECT 이름 FROM 고객 WHERE 나이 IS NULL;
#3-5. 고객 테이블에서 나이가 이미 입력된 고객의 이름을 검색하세요.
SELECT 이름 FROM 고객 WHERE 나이 IS NOT NULL;
#3-6. 고객 테이블에서 이름, 등급, 나이를 검색하되, 나이를 기준으로 내림차순 정렬하세요.
SELECT 이름, 등금, 나이 FROM 고객
ORDER BY 나이 DESC;
#3-7. 고객 테이블에 고객이 몇 명 등록되어 있는지 검색하세요.
SELECT COUNT(아이디) AS 고객수 FROM 고객;
SELECT COUNT(*) AS 고객수 FROM 고객;
#3-8. 고객 테이블에서 적립금 평균이 1,000원 이상인 등급에 대해 등급별 고객수와 적립금 평균을 검색하세요.
SELECT 등금, COUNT(*) AS 고객수, AVG(적립금) AS 평균적립금 FROM 고객
GROUP BY 등금
HAVING AVG(적립금) >= 1000;

#4. 제품 및 주문 테이블에서 검색
#4-1. 제품 테이블에서 제조업체를 중복을 제거하여 검색하세요.
SELECT DISTINCT 제조업체 FROM 제품;
#4-2. 제품 테이블에서 제품명과 단가를 검색하되, 단가에 500원을 더해 '조정 단가'라는 새 이름으로 검색하세요.
SELECT 제품명, 단가+500 AS '조정 단가' FROM 제품;
#4-3. 제품 테이블에서 한빛제과가 제조한 제품의 제품명, 재고량, 단가를 검색하세요.
SELECT 제품명, 제고량, 단가 FROM 제품
WHERE 제조업체 = '한빛제과';
#4-4. 주문 테이블에서 apple 고객이 15개 이상 주문한 주문제품, 수량, 주문일자를 검색하세요.
SELECT 주문제품, 수량, 주문일자 FROM 주문
WHERE 주문고객 = 'apple' AND 수량 >= 15;
#4-5. 제품 테이블에서 모든 제품의 단가 평균을 검색하세요.
SELECT AVG(단가) FROM 제품;
#4-6. 제조업체 별로 제품의 개수와 가장 비싼 단가를 검색하세요.
SELECT 제조업체, COUNT(*) AS '제품수', MAX(단가) AS 최고가
FROM 제품
GROUP BY 제조업체;
SELECT 제조업체, COUNT(*) AS '제품수', MAX(단가) AS 최고가
FROM 제품
GROUP BY 제조업체
HAVING COUNT(*) >= 3;
#5. 외래키를 이용하여 검색
#5-1. banana 고객이 주문한 제품의 이름을 검색하세요.
SELECT 제품.제품명
FROM 제품, 주문
WHERE 주문.주문고객 = 'banana' AND 주문.주문제품 = 제품.제품번호;
#5-2. 나이가 30세 이상인 고객이 주문한 제품의 번호와 주문일자를 검색하세요.
SELECT 주문.주문제품, 주문.주문일자
FROM 고객, 주문
WHERE 고객.나이 >= 30 AND 고객.아이디 = 주문.주문고객;
#5-3. 제조업체 별 총 주문 수량을 검색하세요. (제품 상관없음)
SELECT 제품.제조업체, SUM(주문.수량) AS 주문수량
FROM 제품, 주문
WHERE 주문.주문제품 = 제품.제품번호
GROUP BY 제품.제조업체;
#5-4. 고객 별 주문 총 수량을 검색하세요.
SELECT 고객.이름, SUM(주문.수량) AS 주문수량
FROM 고객, 주문
WHERE 고객.아이디 = 주문.주문고객
GROUP BY 고객.이름;
#5-5. 가격이 4500 원인 제품을 주문한 고객의 이름과 제품명을 검색하세요.
SELECT 고객.이름, 제품.제품명
FROM 고객, 주문, 제품
WHERE 고객.아이디 = 주문.주문고객 AND 주문.주문제품=제품.제품번호 AND 제품.단가 = 4500;
#5-6. 고객의 이름과 고객이 주문한 제품의 이름을 검색하세요. - 고객의 이름을 정렬하여 출력할 것
SELECT 고객.이름, 제품.제품명
FROM 고객, 주문, 제품
WHERE 고객.아이디 = 주문.주문고객 AND 주문.주문제품 = 제품.제품번호
ORDER BY 고객.이름;
#5-7. 고객 정보 및 주문 정보에 관한 데이터를 고객별로 정렬하여 출력하세요.
SELECT *
FROM 고객, 주문, 제품
WHERE 고객.아이디 = 주문.주문고객 AND 주문.주문제품 = 제품.제품번호
ORDER BY 고객.이름