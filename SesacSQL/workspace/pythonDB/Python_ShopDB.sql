CREATE DATABASE ShopDB;
USE ShopDB;
-- 테이블생성하기 ❖ 'Product' 테이블생성
 CREATE TABLE Product(
	pCode		VARCHAR(10),
	pName		VARCHAR(20),
	price		INT,
	amount	INT
 );
 -- 데이터추가하기 ❖ 'Product' 테이블에데이터추가
 INSERT INTO Product (pCode,pName,price,amount)
 VALUES
('p0001','notebook',1000000,3),
('p0002','mouse',10000,20),
('p0003','keybord',30000,15),
('p0004','moniter',500000,10);
SELECT * FROM Product;
-- Python과 연동할 DB작업