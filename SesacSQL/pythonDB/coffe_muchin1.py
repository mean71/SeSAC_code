import pymysql

conn = pymysql.connect(	# 연결자= pymysql.connect(옵션)
  host='localhost',
  user='root',
  password='root',
  db='ShopDB',
  charset='utf8'
  )
curs = conn.cursor()


coffe1_count = 40
coffe1_price = 300
coffe2_count = 60
coffe2_price = 600
coffe3_count = 20
coffe3_price = 1500
잔액=
거스름돈=
소지금액=

def coffe():
    pass

print('커피 가격 : 300원')
print(f'커피 수량 : {coffe_count}개')

while coffe_count != 0
input('돈을 넣어주세요 : ')
거스름돈 { money - price}을 주고 커피를 주고 커피재고에서 뺀다.
돈이 모자라다며 돈을 다시 반환한다.
거스름돈이 떨어지면 더이상 커피를 팔지않고 돈을 넣는족족 반환한다
def:
  pass
def:
  pass
def:
  pass

def main():
count_buy = input('버튼을 눌러주세요')
print('1: 커피 2: 고급커피 3:말차라떼 4:잔돈반환')

# Initialize coffee types and their prices
coffee_types = {
    1: {"name": "Coffee", "count": 40, "price": 300},
    2: {"name": "Premium Coffee", "count": 60, "price": 600},
    3: {"name": "Matcha Latte", "count": 20, "price": 1500},
}




conn.close()

















남은거스름돈 = ?
total_amount_held = ?

def print_menu():
    print('Available Coffee:')
    for key, coffee in coffee_types.items():
        print(f'{key}: {coffee["name"]} - {coffee["price"]} won ({coffee["count"]} left)')

def buy_coffee(coffee_type, money):
    global machine_change
    global total_amount_held

    coffee = coffee_types[coffee_type]
    price = coffee["price"]
    
    if coffee["count"] <= 0:
        print(f'Sorry, {coffee["name"]} is out of stock.')
        return money
    
    if money < price:
        print('Insufficient funds. Returning your money.')
        return money

    if machine_change < (money - price):
        print('Not enough change in the machine. Returning your money.')
        return money
    
    coffee["count"] -= 1
    change = money - price
    machine_change -= change
    total_amount_held += price

    print(f'Dispensing {coffee["name"]}. Your change is {change} won.')
    return change

def main():
    global machine_change

    while True:
        print_menu()
        try:
            money = int(input('Please enter money: '))
            selection = int(input('Please press the button (1: Coffee, 2: Premium Coffee, 3: Matcha Latte, 4: Return): '))
            
            if selection == 4:
                print(f'Returning your money: {money} won.')
                continue
            
            if selection not in coffee_types:
                print('Invalid selection. Please choose a valid option.')
                continue

            change = buy_coffee(selection, money)
            if change != money:
                print(f'Your remaining change: {change} won')
        except ValueError:
            print('Please enter a valid number.')

        if all(coffee["count"] == 0 for coffee in coffee_types.values()):
            print('All coffee types are out of stock. The machine is shutting down.')
            break