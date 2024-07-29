CREATE DATABASE playerDB;
USE playerDB;
CREATE TABLE player(
    floor INT,
    name VARCHAR(40),
    type VARCHAR(40)
 );
INSERT INTO player (floor, name, type)
VALUES
(1, '롯데월드몰', '쇼핑'),
(5, '롯데시네마', '영화관'),
(76, '시그니엘 서울', '호텔'),
(118, '서울 스카이 전망대', '전망대'),
(123, '서울 스카이 전망대', '전망대');
SELECT * FROM player