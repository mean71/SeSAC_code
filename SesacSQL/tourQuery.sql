CREATE DATABASE tourDB;
USE tourDB;

-- tourTable 테이블 데이터 조회하기
SELECT * FROM tourTable;

-- 시도명 중복 제거1 
SELECT DISTINCT 시도명 FROM tourTable;

-- 시도명 중복 제거2
SELECT 시도명 FROM tourTable
GROUP BY 시도명;

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

-- 가장 인기가 없는 관광지부터 인기가 많은 순으로 조회
SELECT * FROM tourTable
ORDER BY 검색건수 ASC;

-- 10개의 관광지 정보만 조회
SELECT * FROM tourTable
LIMIT 10;
