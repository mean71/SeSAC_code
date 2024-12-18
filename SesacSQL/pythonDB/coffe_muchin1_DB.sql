CREATE DATABASE coffe_muchinDB;
USE coffe_muchinDB;
CREATE TABLE muchin_config(
	money			INT,
    beatcoin		INT,
    coffe1_count	INT,
	coffe2_count	INT,
    coffe3_count	INT,
	coffe1_price	INT,
    coffe2_price	INT,    
    coffe3_price	INT
 );
INSERT INTO Product (money, beatcoin, coffe1_count, coffe2_count, coffe3_count, coffe1_price, coffe2_price, coffe3_price)
VALUES
(40000, 20000, 34, 78, 30, 300, 600, 1500);
SELECT * FROM muchin_config;