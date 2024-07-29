game_status = {'x_positions' : [], 'o_positions' : []}
x_positions = game_status['x_positions']
o_positions = game_status['o_positions']
xsize,ysize = 3,3

def display_board(x_positions, o_positions):
    """Create an empty board. 
    The board is made of horizontal lines, made with - and vertical lines, made with |. 
    (optional) When there are no x_cell_size and y_cell_size arguments, default to arbitary size of your choice. Just make it consistent. 
    """
    '''
    빈 보드를 -수평선과 |수직선으로 구성.(선택사항) x_cell_size , y_cell_size 인수 없는 경우 기본값은 원하는 임의크기입니다. 일관성을 유지.
    '''
    board = ''
    h = ' ---------'
    v = '|         '
    O = '|    o    '
    X = '|    x    '
    
    for y in range(ysize):
        board += h*xsize +'\n' + v*xsize + '|\n'
        for x in range(xsize):
            if (x,y) in x_positions:
                board += X 
            elif (x,y) in o_positions:
                board += O
            else:
                board += v
        board += '\n'            
        board += v*xsize + '|\n'
    board += h*xsize
    print(board)

def play(game_status, x_or_o, move):
    if x_or_o == 'X':
        game_status['x_positions'].append(move)
    elif x_or_o == 'O':
        game_status['o_positions'].append(move)
    pass
    """Main function for simulating tictactoe game moves. 
    Tictactoe game is executed by two player's moves. In each move, each player chooses the coordinate to place their mark. It is impossible to place the mark on already taken position. 
    A move in the tictactoe game is composed of two components; whether who ('X' or 'O') made the move, and how the move is made - the coordinate of the move. 
    Coordinate in our tictactoe system will use the coordinate system illustrated in the example below. 
    Example 1. 3 * 4 tictactoe board.
    틱택토 동작 시뮬레이션 기능. 게임은 두 플레이어의 움직임으로 실행. 각 이동에서 플레이어는 표시 배치좌표를 선택. 이미 된 위치에 배치불가능.
    동작은 두 가지 요소로 구성. 1. 누가('X' | 'O') 이동했는지, 이동방법은 이동좌표. 시스템의 좌표는 아래 좌표시스템.
    넣을 좌표를 선택하고 x를 넣었다가 넣을 좌표를 선택하고 o를 넣었다가 반복
 예시 1. 3*4 틱택토 보드.
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

def check_winlose(game_status):
    """Check the game status; game status should be one of 'X wins', 'O wins', 'tie', 'not decided'."""
    winingposition = [
        [(0,0),(0,1),(0,2)],
        [(1,0),(1,1),(1,2)],
        [(2,0),(2,1),(2,2)],
        [(0,0),(1,0),(2,0)],
        [(0,1),(1,1),(2,1)],
        [(0,2),(1,2),(1,2)],
        [(0,0),(1,1),(2,2)],
        [(0,2),(1,1),(2,0)],  
    ]
    if winingposition in x_positions:
        print('X Winnnnn!')
        return 'x win'
    elif winingposition in o_positions:
        print ('O Winnnnn!')
        return 'o win'
    elif len(x_positions+o_positions) == 9:
        print('ox_compare')
        return 'ox_compare'
    else:
        return 'not decided'

def display(game_status):
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
    보드의 현재 스냅샷을 표시합니다.'스냅샷'에는 다음 구성 요소가 포함되어야 합니다.
    - 보드 자체
    - 이미 이루어진 동작
    명확한 설명은 제공된 예를 참조하세요.
    예시 1.
    TictactoeGame 인스턴스에 다음 속성이 있는 경우;
    - x_위치 = [(0,0), (2,0), (2,1), (1,2)]
    - o_positions = [(1,0), (1,1), (0,2), (2,2)]
    t.디스플레이()
    """
    display_board(game_status['x_positons'], game_status['o_positions'])
#   pass 
