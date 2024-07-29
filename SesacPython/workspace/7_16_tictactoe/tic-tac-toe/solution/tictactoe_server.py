try:
    from solution.tictactoe import TictactoeGame
except ImportError:
    from tictactoe import TictactoeGame
        
def play_game(game, x_player, o_player):
    x_or_o = 'X'
    
    while game.check_winlose() == 'not decided':
        print('==============')
        x_positions = game.x_positions
        o_positions = game.o_positions
        
        if x_or_o == 'X':
            x_move = x_player(x_or_o, x_positions, o_positions)
            game.play(x_or_o, x_move)
            x_or_o = 'O'
            print(f'x_player moved to {x_move}')
        else:
            o_move = o_player(x_or_o, x_positions, o_positions)
            game.play(x_or_o, o_move)
            x_or_o = 'X'
            print(f'o_player moved to {o_move}')
        game.display()
    print(game.check_winlose())
    return game.check_winlose()

if __name__ == '__main__':
    from player import random_player 
    game = TictactoeGame()
    play_game(game, random_player, random_player)