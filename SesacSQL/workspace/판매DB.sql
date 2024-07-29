-- 문제 1-1
CREATE TABLE 고객(
    아이디 	VARCHAR(20) 	PRIMARY KEY,
    이름 	VARCHAR(10) 	NOT NULL, 
    나이 	INT,
    등급 	VARCHAR(10) 	NOT NULL,
    직업 	VARCHAR(20),
    적립금 	INT 			DEFAULT 0
 );
CREATE TABLE 제품(
	제품번호	CHAR(3) 		PRIMARY KEY,
    제품명	VARCHAR(20),
    재고량	INT,
    단가		INT,
    제조업체	VARCHAR(20)
);
CREATE TABLE 주문 (
	주문번호 	CHAR(3) 		PRIMARY KEY,
    주문고객 	VARCHAR(20),
    주문제품 	CHAR(3),
    수량		INT,
    배송지	VARCHAR(30),
    주문일자 	DATE,
    FOREIGN KEY(주문고객) REFERENCES 고객(아이디),
    FOREIGN KEY(주문제품) REFERENCES 제품(제품번호)
);
-- 3-1
SELECT 아이디, 이름, 등급 FROM 고객;
-- 3-2
SELECT 이름, 나이, 등급, 적립금 FROM 고객 WHERE 이름 LIKE '김%';
-- 3-3
SELECT 아이디, 이름, 등급 FROM 고객 WHERE 아이디 LIKE '_____';
-- 3-4
SELECT 이름 FROM 고객 WHERE 나이 IS NULL;
-- 3-5
SELECT 이름 FROM 고객 WHERE 나이 IS NOT NULL;
-- 3-6
SELECT 이름, 등급, 나이 FROM 고객 
ORDER BY 나이 DESC;
-- 3-7
SELECT COUNT(아이디) AS 고객수 FROM 고객;
SELECT COUNT(*) AS 고객수 FROM 고객;
-- 3-8
SELECT 등급, COUNT(*) AS 고객수, AVG(적립금) AS 평균적립금 FROM 고객
GROUP BY 등급 
HAVING AVG(적립금) >= 1000;
-- 4-1
SELECT DISTINCT 제조업체 FROM 제품;
-- 4-2
SELECT 제품명, 단가+500 AS '조정 단가' FROM 제품;
-- 4-3
SELECT 제품명, 재고량, 단가 FROM 제품 
WHERE 제조업체 = '한빛제과';
-- 4-4
SELECT 주문제품, 수량, 주문일자 FROM 주문
WHERE 주문고객 = 'apple' AND 수량 >= 15;
-- 4-5
SELECT AVG(단가) as '단가' FROM 제품;
-- 4-6
SELECT 제조업체, COUNT(*) AS '제품수', MAX(단가) AS 최고가
FROM 제품
GROUP BY 제조업체;
-- 4-7
SELECT 제조업체, COUNT(*) AS '제품수', MAX(단가) AS 최고가
FROM 제품
GROUP BY 제조업체
HAVING COUNT(*) >= 3;
-- 5-1
SELECT 제품.제품명
FROM 제품, 주문
WHERE 주문.주문고객 = 'banana' AND 주문.주문제품 = 제품.제품번호;
-- 5-2
SELECT 주문.주문제품, 주문.주문일자
FROM 고객, 주문
WHERE 고객.나이 >= 30 AND 고객.아이디 = 주문.주문고객;
-- 5-3
SELECT 제품.제조업체, SUM(주문.수량) AS 주문수량
FROM 제품, 주문
WHERE 주문.주문제품 = 제품.제품번호
GROUP BY 제품.제조업체;
-- 5-4
SELECT 고객.이름, SUM(주문.수량) AS 주문수량
FROM 고객, 주문
WHERE 고객.아이디 = 주문.주문고객
GROUP BY 고객.이름;
-- 5-5
SELECT 고객.이름, 제품.제품명
FROM 고객, 주문, 제품
WHERE 고객.아이디 = 주문.주문고객 AND 주문.주문제품=제품.제품번호 AND 제품.단가 = 4500;
-- 5-6
SELECT 고객.이름, 제품.제품명
FROM 고객, 주문, 제품
WHERE 고객.아이디 = 주문.주문고객 AND 주문.주문제품 = 제품.제품번호
ORDER BY 고객.이름;
-- 5-7
SELECT *
FROM 고객, 주문, 제품
WHERE 고객.아이디 = 주문.주문고객 AND 주문.주문제품 = 제품.제품번호
ORDER BY 고객.이름

