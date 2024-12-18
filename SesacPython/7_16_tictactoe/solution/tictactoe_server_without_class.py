from tictactoe_without_class import check_winlose, play, display 

def play_game(x_player, o_player):
    game_status = {'x_positions' : [], 'o_positions' : []} 
    x_or_o = 'X'
    
    while check_game_winlose(game_status) == 'not decided':
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
    print(check_game_winlose(game_status))
    return game_status 

if __name__ == '__main__':
    from player import random_player 
    game = TictactoeGame()
    play_game(game, random_player, random_player)