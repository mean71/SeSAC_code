import pymysql

conn = pymysql.connect(	# 연결자= pymysql.connect(옵션)
  host='localhost',
  user='root',
  password='root',
  db='ShopDB',
  charset='utf8'
  )
curs = conn.cursor()


def
  두플레이어의 번호를 비교해서 승패결과계산
 
def
mine = "가위"
yours = "바위"

if mine == yours:
    print("비김")
    
elif mine == "가위" and yours == "보":
    print("이겼다")
    
elif mine == "바위" and uours == "가위":
    print("이겼다")
    
elif mine == "보" and yours == "바위":
    print("이겼다")
else:
    print("졌다")

def

def

def main():
  print('1.가위! 2.바위! 3.보! : ')
  conn.close()


