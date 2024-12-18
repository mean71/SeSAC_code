import random 

small_max_num = 10  # 작은 최대치 10
max_num = 10000000  # 숫자 최대치 천만
small_random_int = random.randint(0, small_max_num)     # 0이상small_max이하 랜덤정수하나 생성
random_int = random.randint(0, max_num)                 # 0이상 max_num이하 랜덤정수하나 생성
random_float = random.uniform(0, max_num)               # 0이상 max_num이하 랜덤실수하나 생성
#guess : 추측하다
def updown_game_easy(guess):                        # easy난이도 업다운 # 인수가 10단위난수보다 큰지작은지 같으면 인수값 반환
    if guess > small_random_int:
        return 'down'
    elif guess < small_random_int:
        return 'up'
    return guess == small_random_int

def updown_game_medium(guess):                      # medium난이도 업다운 # 인수가 천만범위정수난수보다 큰지작은지 == 인수값반환
    if guess > random_int:
        return 'down'
    elif guess < random_int:
        return 'up'
    return guess == random_int

def updown_game_hard(guess):                        # hard 난이도 # 인수가 0.001<
    if guess - random_float > 0.001:
        return 'down'
    elif guess - random_float < -0.001:
        return 'up' 
    return abs(random_float - guess) < 0.001


if __name__ == '__main__':
    from finder import manual_finder, naive_finder, smart_finder
    from time import time 

    begin = time()
    print(manual_finder(updown_game_easy))
    end = time()
    
    begin = time()
    print(naive_finder(updown_game_medium, list(range(5))))
    end = time()
    
    begin = time()
    print(smart_finder(updown_game_hard))
    end = time()
    
    print(end - begin)
'''
def manual_finder(f):
    while True:
        i = input(f'Guess the argument!\nGuess is: ')
        res = f(float(i))
        if res is True:
            print(f'You found the right argument!; {float(i)}')
            return float(i)
        print('manual_finder실행')
        print(res)
        
# f의 절반을 업다운에 돌리고 리턴이 업이면 위의 절반 다운이면 아래의 절반
def naive_finder(f, lst = list(range(5))):
    lst
    f
    while True:
        res = f(float(?))
        if res is True:
            print(f'You found the right argument!; {float(i)}')
            return float(i)
        print('manual_finder실행')
        print(res)
    pass

def smart_finder(f, min_input = 0, max_input = 100):

    pass
'''