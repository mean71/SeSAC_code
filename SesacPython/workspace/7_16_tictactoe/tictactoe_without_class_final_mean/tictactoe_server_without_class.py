from tictactoe_without_class import check_winlose, play, display_board
from player import random_player

x_or_o = 'X'                                            # x_or_o값을 X로 시작

def play_game(x_or_o):
    game_status = {'x_positions' : [], 'o_positions' : []}  
    
    while check_winlose(game_status) == 'not decided':  # 승패체크(game_status)돌려서 승부가 안난경우 무한반복
        print('==============')                         # 턴 구분선
        x_positions = game_status['x_positions']        # x포지션에 겜스테이터스의 x리스트를 대입
        o_positions = game_status['o_positions']        # o포지션에 겜스테이터스의 o리스트를 대입
        
        if x_or_o == 'X':                                       # if X차례:
            x_move = random_player(x_positions, o_positions)     # x_move에 random_player반환값 대입
            play(game_status, x_or_o, x_move)                       # paly(대입한x리스트 , X , random_player반환값)
            x_or_o = 'O'                                            # 다음차례 O집어넣기
            print(f'x_player moved to {x_move}')                    # X플레이어는 {x_move}로 이동했습니다 출력
        else:                                                   # else O 차례:
            o_move = random_player(x_positions, o_positions)     # o_move에 random_player반환값 대입
            play(game_status, x_or_o, o_move)                       # paly(대입한x리스트 , Y , random_player반환값)
            x_or_o = 'X'                                            # 다음차례 X를 집어넣기
            print(f'O_player moved to {o_move}')                    # O플레이어는 {o_move}로 이동했습니다 출력
        display_board(game_status['x_positions'],game_status['o_positions']) # display 가져와서 game_status넣고 실행
    print(check_winlose(game_status))                # 승패체크(game_s)함수 반환값 출력하고 겜스테이터스 반환
    return game_status

if __name__ == '__main__':    # 이게 없는 코드파일을 다른곳에서 import할시 import와 동시에 해당파일의 모든코드를 실행._
    game_status = {'x_positions' : [], 'o_positions' : []}  
    x_positions = game_status['x_positions']        # x포지션에 겜스테이터스의 x리스트를 대입
    o_positions = game_status['o_positions']  
    play_game(x_or_o)    # play_game(X , 랜덤플, 랜덤플)    