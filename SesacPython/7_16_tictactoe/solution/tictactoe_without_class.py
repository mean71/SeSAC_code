game_status = {'x_positions' : [], 'o_positions' : []} 

def empty_board(x_size = 3, y_size = 3, x_cell_size = 5, y_cell_size = 3):
    """Create an empty board. 

    The board is made of horizontal lines, made with - and vertical lines, made with |. 

    (optional) When there are no x_cell_size and y_cell_size arguments, default to arbitary size of your choice. Just make it consistent. 

     ---------- ---------- ----------
    |          |          |          |
    |    X     |    O     |    X     |
    |          |          |          |
     ---------- ---------- ----------
    |          |          |          |
    |          |    O     |    X     |
    |          |          |          |
     ---------- ---------- ----------
    |          |          |          |
    |    O     |    X     |    O     |
    |          |          |          |
     ---------- ---------- ----------
    """
    hline =  (' ' + '-' * x_cell_size ) * x_size

    for y in range(y_size):
        print(hline)
        for z in range(y_cell_size):
            for x in range(x_size):
                print('|' + ' ' * x_cell_size, end = '')
            print('|')
    print(hline)    
    
def play(game_status, x_or_o, coordinate):
    """Main function for simulating tictactoe game moves. 

    Tictactoe game is executed by two player's moves. In each move, each player chooses the coordinate to place their mark. It is impossible to place the mark on already taken position. 

    A move in the tictactoe game is composed of two components; whether who ('X' or 'O') made the move, and how the move is made - the coordinate of the move. 

    Coordinate in our tictactoe system will use the coordinate system illustrated in the example below. 
    
    Example 1. 3 * 4 tictactoe board. 
    
         ---------- ---------- ----------
        |          |          |          |
        |  (0,0)   |  (1,0)   |  (2,0)   |
        |          |          |          |
         ---------- ---------- ----------
        |          |          |          |
        |  (0,1)   |  (1,1)   |  (2,1)   |
        |          |          |          |
         ---------- ---------- ----------
        |          |          |          |
        |  (0,2)   |  (1,2)   |  (2,2)   |
        |          |          |          |
         ---------- ---------- ----------
        |          |          |          |
        |  (0,3)   |  (1,3)   |  (2,3)   |
        |          |          |          |
         ---------- ---------- ----------
        """
    
    # game_status['x_positions']
    # 이때까지 X가 놓인 위치들 
    # game_status['o_positions']
    # 이때까지 O가 놓인 위치들 

    # game_status['x_positions'] + game_status['o_positions']
    # 이때까지 X가 놓인 위치들  + 이때까지 O가 놓인 위치들 
    # 이때까지 X 아니면 O가 놓인 위치들 

    # play(game_status, x_or_o, coordinate)
    # 이번에 x_or_o를 coordinate에 두었다. 
    # ex) x_or_o = 'X', coordinate = (0,0)
    # X를 (0,0)에 두었다. --> X가 놓인 걸 기록하고 있는 곳에다가, (0,0)을 추가하면 됨 
    # game_status['x_positions'].append(coordinate)
    # list l에 1을 추가함 --> l.append(1)

        # ---------- ---------- ----------
        # |          |          |          |
        # |  X       |  (1,0)   |  (2,0)   |
        # |          |          |          |
        #  ---------- ---------- ----------
        # |          |          |          |
        # |  (0,1)   |  (1,1)   |  (2,1)   |
        # |          |          |          |
        #  ---------- ---------- ----------
        # |          |          |          |
        # |  (0,2)   |  (1,2)   |  (2,2)   |
        # |          |          |          |
        #  ---------- ---------- ----------
        # |          |          |          |
        # |  (0,3)   |  (1,3)   |  (2,3)   |
        # |          |          |          |


    if coordinate in game_status['x_positions'] + game_status['o_positions']:
        assert False

    if x_or_o == 'X':
        game_status['x_positions'].append(coordinate) 
    elif x_or_o == 'O':
        game_status['o_positions'].append(coordinate) 
    else:
        raise ValueError(f'x_or_o should be one of X or O; got {x_or_o}')

def check_winlose(game_status):
    """Check the game status; game status should be one of 'X wins', 'O wins', 'tie', 'not decided'. 
    """
    winning_positions = [
        [(0,0), (1,0), (2,0)],  
        [(0,1), (1,1), (2,1)], 
        [(0,2), (1,2), (2,2)], 
        [(0,0), (0,1), (0,2)], 
        [(1,0), (1,1), (1,2)], 
        [(2,0), (2,1), (2,2)], 
        [(0,0), (1,1), (2,2)], 
        [(2,0), (1,1), (0,2)],
    ]

    if determine_if_x_wins(game_status, winning_positions):
        return 'X wins'
    elif determine_if_o_wins(game_status, winning_positions):
        return 'O wins'
    elif len(game_status['x_positions'] + game_status['o_positions']) == 9:
        return 'tie' 
    else:
        return 'undecided' 

def determine_if_x_wins(game_status, winning_positions):
    x_pos = game_status['x_positions']
    for win in winning_positions:
        a, b, c = win 
        if a in x_pos and b in x_pos and c in x_pos:
            return True
    return False 

def determine_if_o_wins(game_status, winning_positions):
    o_pos = game_status['x_positions']
    for win in winning_positions:
        a, b, c = win 
        if a in o_pos and b in o_pos and c in o_pos:
            return True
    return False 

def display(game_status, x_size = 3, y_size = 3, x_cell_size = 4, y_cell_size = 3):
    """Display the current snapshot of the board. 

    'Snapshot' should contain following components. 

    - The board itself 
    - Moves that are already made

    For clarification, see provided examples. 

    Example 1. 
    When TictactoeGame instance t have following attributes; 
    - x_positions = [(0,0), (2,0), (2,1), (1,2)]
    - o_positions = [(1,0), (1,1), (0,2), (2,2)]

    t.display()
    >> 
     ---------- ---------- ----------
    |          |          |          |
    |    X     |    O     |    X     |
    |          |          |          |
     ---------- ---------- ----------
    |          |          |          |
    |          |    O     |    X     |
    |          |          |          |
     ---------- ---------- ----------
    |          |          |          |
    |    O     |    X     |    O     |
    |          |          |          |
     ---------- ---------- ----------

    """
    hline =  (' ' + '-' * x_cell_size ) * x_size

    for y in range(y_size):
        print(hline)
        for z in range(y_cell_size):
            for x in range(x_size):
                if z == 1:
                    left_margin = (x_cell_size) // 2 
                    right_margin = (x_cell_size-1) // 2 
                    if (x, y) in game_status['x_positions']:
                        print('|' + ' ' * left_margin + 'X' + ' ' * right_margin, end = '')
                    elif (x, y) in game_status['o_positions']:
                        print('|' + ' ' * left_margin + 'O' + ' ' * right_margin, end = '')
                    else:
                        print('|' + ' ' * x_cell_size, end = '')
                else:
                    print('|' + ' ' * x_cell_size, end = '')
            print('|')
    print(hline)    


if __name__ == '__main__':
    # empty_board(x_size = 3, y_size = 3, x_cell_size = 5, y_cell_size = 3)
    game_status = {'x_positions' : [(0,0)], 'o_positions' : [(1,0)]} 

    display(game_status)