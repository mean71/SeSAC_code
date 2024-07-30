import pymysql

conn = pymysql.connect(	# 연결자= pymysql.connect(옵션)
  host='localhost',
  user='root',
  password='root',
  db='ShopDB',
  charset='utf8'
  )
curs = conn.cursor()


def 층별안내():
  print("<<1.층별 안내 기능 실행>>")
  try:
    floor = int(input('층 :'))
    sql = 'select * from floors where floor = %s'
    curs.execute(sql, (floor,))
    result = curs.fetchall()
    if result:
      for row in result:
        print(row)
  except Exception as a1:
    print('searchERROR',a1)
  
def 이름검색(name):
  print("<<2.이름검색>>")
  try:
    name = input('시설명 :')
    sql = 'select * from floors where name = %s'
    curs.execute(sql, (name,))
    result = curs.fetchall()
    if result:
      print(result)
    else:
      print(f'{name} 시설이 없습니다.')
  except Exception as a2:
    print('searchERROR',a2)

def 시설타입검색():
  print("<<3.시설타입 검색>>")
  try:
    type = input('시설타입 :')
    sql = 'select * from floors where type = %s'
    curs.execute(sql, (type,))
    result = curs.fetchall()
    if result:
      print(result)
    else:
      print(f'{type} 데이터가 없습니다.')
  except Exception as a3:
    print('TypeSearchERROR',a3)

def 시설추가():
  print("<<4.시설 데이터 추가>>")
  try:
    add_floor = int(input('층 :'))
    add_name = input('시설명 :')
    add_type = input('시설타입 :')
    
    sql = 'insert into floors (floor, name, type) values (%s,%s,%s)'
    curs.execute(sql, (add_floor, add_name, add_type))
    conn.commit()
    
    sql = 'select * from floors where name = %d'
    curs.execute(sql, add_name)
    add_data = curs.fetchone()
    print(f'{add_data}데이터가 추가되었습니다.')
  except Exception as a4:
    print('addERROR',a4)

def 시설수정():
  print("<<5.시설 데이터 수정>>")
  try:
    old_name = input('수정할 시설명 : ')
    update_floor = int(input('새 층 : '))
    update_name = input('새 시설명 : ')
    update_type = input('새 시설타입 : ')
    sql = 'UPDATE floors SET floor = %s, name = %s, type = %s WHERE name = %s'
    curs.execute(sql, (update_floor, update_name, update_type, old_name))
    conn.commit()
    print(f'{old_name} 시설이 수정되었습니다.')
  except Exception as a5:
    print('updateERROR',a5)
  pass

def 시설삭제():
  print("<<6.시설 데이터 삭제>>")
  try:    
    del_name = input('삭제할 시설명 : ')
    sql = 'delete from floors where name = %s'
    curs.execute(sql, del_name)
    conn.commit()
    print(f'"{del_name}"데이터가 삭제 되었습니다.')
  except Exception as a6:
    print('deleteERROR', a6)
  pass

def 전체출력():
  print("<<7.전체 데이터 출력>>")
  try:
    print("<<전층 안내>>")
    sql = "select * from floors"
    curs.execute(sql)
    result = curs.fetchall() 
    for data in result:
      print(data)
  except Exception as a7:
    print('menu7ERROR',a7)
  pass

def main():
  try:
    while True:
      print('----------------------------------------------------------------------------------------------')
      print('1.층별안내\n2.이름검색\n3.타입검색\n4.시설추가\n5.시설수정\n6.시설삭제\n7.전층안내\n8.종료')
      print('----------------------------------------------------------------------------------------------')
      menu = int(input("1.층별안내선택할 기능의 번호를 입력하세요 : "))
      if menu == 1:
        floor = input("2.검색할 층수을 입력하세요: ")
        층별안내()
      elif menu == 2:
        name = input("3.타입검색\n이름검색검색할 이름을 입력하세요: ")
        이름검색(name)
      elif menu == 3:
        name = input("4.시설추가\n검색할 타입을 입력해주세요: ")
        시설타입검색()
      elif menu == 4:
        name = input("5.시설수정\n검색할 이름을 입력하세요: ")
        시설추가()
      elif menu == 5:
        name = input("6.시설삭제\n검색할 이름을 입력하세요: ")
        시설수정()
      elif menu == 6:
        name = input("7.전체출력\n검색할 이름을 입력하세요: ")
        시설삭제()
      elif menu == 7:
        name = input("8.종료검색\n할 이름을 입력하세요: ")
        전체출력()
      elif menu == 8:
        print('------------------------------------------------------------------------------------------------------------------\n시스템 종료\n')
        break
      else:
        print("잘못된 선택입니다.")
  except Exception as M:
    print(f"오류가 발생했습니다: {M}")
    conn.close()