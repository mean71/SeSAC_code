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
    print()
    
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
    pass 

def check_winlose(game_status):
    """Check the game status; game status should be one of 'X wins', 'O wins', 'tie', 'not decided'. 
    """
    return '' 


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
    print()

if __name__ == '__main__':
    # empty_board(x_size = 3, y_size = 3, x_cell_size = 5, y_cell_size = 3)
    game_status = {'x_positions' : [(0,0)], 'o_positions' : [(1,0)]} 

    display(game_status)