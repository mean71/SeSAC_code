from tictactoe_without_class import check_winlose, play, display 
from time import sleep 

def play_game(x_player, o_player):
    game_status = {'x_positions' : [], 'o_positions' : []} 
    x_or_o = 'X'
    
    while check_winlose(game_status) == 'not decided':
        print('==============')
        x_positions = game_status['x_positions']
        o_positions = game_status['o_positions']
        
        if x_or_o == 'X':
            x_move = x_player(x_or_o, x_positions, o_positions)
            play(game_status, x_or_o, x_move)
            x_or_o = 'O'
            print(f'x_player moved to {x_move}')
        else:
            o_move = o_player(x_or_o, x_positions, o_positions)
            play(game_status, x_or_o, o_move)
            x_or_o = 'X'
            print(f'o_player moved to {o_move}')
        display(game_status)
        sleep(1)
    print(check_winlose(game_status))
    return game_status 

if __name__ == '__main__':
    from player import random_player
    from player_mj import smart_player as mj
    from player_loser import smart_player as ls 
    # play_game(mj, ls)
    play_game(ls, mj)