# class 울트라틱택토:
#   플레이어1,2
#   9*9칸의 틱택토 보드좌표 [[길의9의리스트]를 길이9의 리스트로 나열]
  
#   def 보드판의 내용을 디스플레이에 시각화:
#   def 플레이어가 좌표를 입력하면 해당 좌표값을 공백에서 플레이어심볼로 교체 다음에 실행할 플레이어값 변환:
#   def 9*9칸중 한 칸에 좌표가 입력되면 다음차례에 입력받을 빙고칸을 계산해서 반환:
#   def 입력받은 빙고칸에서 빙고달성여부계산:
#   def :
class Ultra Tic Tac Toe:
  def __init__(self):
    self.board = [[' ' for _ in range(9)] for _ in range(9)]
    self.current_player = 'X'
    self.next_board = None

  def display_board(self):
    for row in self.board:
      print('|'.join(row))
      print('-' * 17)

  def make_move(self, board_row, board_col, cell_row, cell_col):
    if self.board[board_row * 3 + cell_row][board_col * 3 + cell_col] == ' ':
      self.board[board_row * 3 + cell_row][board_col * 3 + cell_col] = self.current_player
      self.next_board = (cell_row, cell_col)
      self.current_player = 'O' if self.current_player == 'X' else 'X'
      return True
    return False

  def calculate_next_board(self, cell_row, cell_col):
    return cell_row, cell_col

  def check_winner(self, board_row, board_col):
    start_row, start_col = board_row * 3, board_col * 3
    for i in range(3):
      if self.board[start_row + i][start_col] == self.board[start_row + i][start_col + 1] == self.board[start_row + i][start_col + 2] != ' ':
        return True
      if self.board[start_row][start_col + i] == self.board[start_row + 1][start_col + i] == self.board[start_row + 2][start_col + i] != ' ':
        return True
    if self.board[start_row][start_col] == self.board[start_row + 1][start_col + 1] == self.board[start_row + 2][start_col + 2] != ' ':
      return True
    if self.board[start_row][start_col + 2] == self.board[start_row + 1][start_col + 1] == self.board[start_row + 2][start_col] != ' ':
      return True
    return False

  def play(self):
    while True:
      self.display_board()
      if self.next_board:
        print(f"Next board to play: {self.next_board}")
      board_row = int(input("Enter board row (0-2): "))
      board_col = int(input("Enter board col (0-2): "))
      cell_row = int(input("Enter cell row (0-2): "))
      cell_col = int(input("Enter cell col (0-2): "))
      if self.make_move(board_row, board_col, cell_row, cell_col):
        if self.check_winner(board_row, board_col):
          self.display_board()
          print(f"Player {self.current_player} wins!")
          break
      else:
        print("Invalid move, try again.")