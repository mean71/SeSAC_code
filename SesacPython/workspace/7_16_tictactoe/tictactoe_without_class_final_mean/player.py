from random import randint 
x_or_o = 'X'

def random_player(x_positions, o_positions):
    move = (0, 0)           # move = (0,0) 으로 생성
    while move in x_positions + o_positions:    # move좌표가 이미 O,X가 둔 x_positions, o_positions 위치에 없다면 실행
        x = randint(0, 2)
        y = randint(0, 2)
        move = (x, y)       # move에 3x3의 랜덤좌표를 대입해서 반환
    return move

def smart_player(x_or_o, x_positions, o_positions): # 이건...
    return random_player(x_or_o, x_positions, o_positions)