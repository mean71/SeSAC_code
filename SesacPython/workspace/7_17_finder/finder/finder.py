"""
Implement a better finder to find the right argument for the function.
Your job is to implement a function that accepts another function(call this f) and additional information(related to possible candidates) as input, and returns the argument that f returns True.
As a hint, f will return 'up' or 'down'. When f needs larger input value to return True, it will return 'up'. Else, it will return 'down'.
You will be asked to implement 2 finder functions; naive_finder and smart_finder.
1) naive_finder
Function naive_finder assumes that the test function only accepts integer inputs; therefore, naive_finder can (naively) iterate all the possible candidates. It will take long - but that's why it's called naive.  Function naive_finder accepts another function f and a candidate list as input. When naive_finder is called, it iterates over all possible candidates, applies all candidates to the function one at a time, and returns when the result is True.
naive_finder should be able to find right argument for updown_game.updown_game_easy and updown_game.updown_game_medium.
2) smart_finder
Function smart_finder accepts another function, and the max/min value of the input for the function f. To implement the smart_finder function, think of how you actually play '업다운 게임'.
smart_finder should be able to find right argument for updown.game.updown_game_hard and animation.check_collision.
"""
'''
함수에 대한 올바른 인수를 찾으려면 더 나은 검색기를 구현하십시오.
당신의 임무는 다른 함수(이 f를 호출)와 추가 정보(가능한 후보와 관련된)를 입력으로 받아들이고
f가 True를 반환하는 인수를 반환하는 함수를 구현하는 것
힌트로 f는 'up' 또는 'down'을 반환합니다. f가 True를 반환하기 위해 더 큰 입력 값이 필요한 경우 'up'반환.
그렇지 않으면 'down'을 반환.
2개의 찾기 기능을 구현하라는 메시지표시. naive_finder 및 smart_finder.
1) 순진한_파인더
naive_finder 함수는 테스트 함수가 정수 입력만 허용한다고 가정합니다.
따라서 naive_finder는 가능한 모든 후보를 (순진하게) 반복할 수 있습니다. 시간이 오래 걸릴 것입니다.
그러나 그것이 순진하다고 불리는 이유입니다. naive_finder 함수는 또 다른 함수 f와 후보 목록을 입력으로 받아들입니다.
naive_finder가 호출되면 가능한 모든 후보를 반복하고 모든 후보를 한 번에 하나씩 함수에 적용하고 결과가 True일 때 반환
naive_finder는 updown_game.updown_game_easy 및 updown_game.updown_game_medium에 대한 바른 인수 찾을 수 있어야.
2) 스마트_파인더
함수 smart_finder는 다른 함수와 함수 f에 대한 입력의 최대/최소 값을 허용합니다.
smart_finder 기능을 구현하려면 '업다운 게임'을 실제로 어떻게 플레이할지 생각.
smart_finder는 updown.game.updown_game_hard 및 animation.check_collision에 대한 올바른 인수를 찾을 수 있어야.
'''
# updown_game.py 용 과제로 전체호출
def manual_finder(f):                   #animation에서 호출
    while True:
        i = input(f'Guess the argument!\nGuess is: ')
        res = f(float(i))
        if res is True:
            print(f'You found the right argument!; {float(i)}')
            return float(i)
        print('manual_finder실행')
        print(res)

def naive_finder(f, lst = list(range(5))):            # get_angle 코드에서 호출
    while True:
        i = input(f'Guess the argument!\nGuess is: ') # 인풋대신
        res = f(float(i))
        if res is True:
            print(f'You found the right argument!; {float(i)}')
            return float(i)
        print('manual_finder실행')
        print(res)
    pass

def smart_finder(f, min_input = 0, max_input = 100):
    
    pass