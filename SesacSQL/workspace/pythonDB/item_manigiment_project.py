import pymysql

conn = pymysql.connect(	# 연결자= pymysql.connect(옵션)
  host='localhost',
  user='root',
  password='root',
  db='ShopDB',
  charset='utf8'
  )
curs = conn.cursor()
  
def item_list():
  try:
    print('<<전체 제품 보기>>')
    sql = "select * from product"
    curs.execute(sql)
    result = curs.fetchall()
    for data in result:
      print(data)
  except:
    print('menu1ERROR')
  pass

def item_search():
  try:
    search_item = input('제품번호입력 :')
    sql = "select * from product where pcode = %s"
    curs.execute(sql, search_item)
    result = curs.fetchone()
    if result:
      print(result)
    else:
      print('찾을 수 없는 코드입니다.')
  except:
    print('searchERROR') 
  pass

def item_add():
  try:
    item_add_code = input('추가할 제품Code 입력 : ')
    item_add_name = input('추가할 제품명 입력 : ')
    item_add_price = input('추가 제품 가격입력 : ')
    item_add_amount = input('추가 제품 재고입력 : ')
    
    sql = "insert into product (pCode, pName, price, amount) values (%s,%s,%s,%s)"
    curs.execute(sql, (item_add_code, item_add_name, item_add_price, item_add_amount))
    conn.commit()
    
    sql = 'select pname from product where pcode = %s'
    curs.execute(sql, item_add_code)
    item_add_name = curs.fetchone()
    print(f'{item_add_name}상품추가되었습니다.')
  except:
    print('addERROR')
  pass

def item_update():
  try:
    update_item = input('수정할 제품코드를 입력해주세요 : ')
    sql = 'select * from product where pcode = %s'
    curs.execute(sql, update_item)
    result = curs.fetchone()
    print()
    item_update_code = input('제품Code 수정 : ')
    item_update_name = input('제품명 수정 : ')
    item_update_price = input('가격 수정 : ')
    item_update_amount = input('재고 수량 : ')
    sql = "update product set pcode = %s, pname = %s, price = %s, amount = %s where pcode = %s"
    curs.execute(sql, (item_update_code, item_update_name, item_update_price, item_update_amount, update_item))
    conn.commit()
    
    sql = 'select pname from product where pcode = %s'
    curs.execute(sql, update_item)
    update_item_name = curs.fetchone()
    print(f'{update_item_name}수정되었습니다.')
    
  except:
    print('updateERROR')
  pass

def item_del():
  try:
    item_del = input('삭제할 제품코드를 입력해주세요 : ')
    sql = 'delete from product where pcode = %s'
    curs.execute(sql, item_del)
    conn.commit()
    
    sql = 'select pname from product where pcode = %s'
    curs.execute(sql, item_del)
    del_item_name = curs.fetchone()
    print(f'"{del_item_name}"제품목록에서 삭제 되었습니다.')
  except:
    print('deleteERROR')
  pass

def main():
  while True:
    try:
      print('#############################################################################################')
      print('                                    제품 관리 프로그램                                        ')
      print('#############################################################################################')
      print('메뉴를 선택하세요 => 전체제품보기(1), 제품검색(2), 제품추가(3), 제품수정(4), 제품삭제(5), 종료(6)')
      select_menu=int(input('숫자입력 : '))
      if select_menu == 1:
        item_list()
      elif select_menu == 2:
        item_search()
      elif select_menu == 3:
        item_add()
      elif select_menu == 4:
        item_update()
      elif select_menu == 5:
        item_del()
      elif select_menu ==6:
        print('<< 프로그램을 종료합니다. >>')
        break
      else:
        print('번호를 입력해주세요.')
    except:
      print("오류가 발생하여 재시작합니다. 메뉴를 선택해주세요")
    finally:
      print('\n\n- - -\n\n')
      conn.close()

main()