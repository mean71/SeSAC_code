import os
game_status = {'x_positions' : [], 'o_positions' : []} 
os.system('chcp 65001')
print("")
def making_board(x_size,y_size):    
    board_list=[]
    for x in range(0,x_size):
        row=[]
        for y in range(0,y_size):
            row.append("0")
        board_list.append(row)
    return board_list

def empty_board():
    board_list=making_board(3,3)
    def drawboard():
        import random
        boardlength=7
        pictograph="☂☏☯☮✈✉☕✌❄⻄⛅⚠♻✨∀☘⌚⏩⏭"
        randompicto = lambda : str(pictograph[random.randint(0,len(pictograph)-1)])

        def rowf():
          for row in range(3):
            for count in range(boardlength):
              print(randompicto(),end="")
          print(randompicto(),"")

        def columnf():
          for column in range(6):
            for count2 in range(4):
              print(randompicto(),"⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",end="")
            print("")
        rowf()
        for i in range(len(board_list)):
            columnf()
            rowf()        
    board = drawboard()
    return board      


empty_board()

def play(game_status, x_or_o, coordinate):
    if coordinate in game_status['x_positions']+game_status['o_positions']:
        print("cant move that way")
    elif x_or_o == "X":
        game_status['x_positions'].append(coordinate)
    elif x_or_o == "O":
        game_status['o_positions'].append(coordinate)
    else:
        print("wrong input")



def check_winlose(game_status):
    x_positions = game_status['x_positions']
    o_positions = game_status['o_positions']
    from collections import Counter
    board_list1=making_board(3,3)
    x_coord = list(game_status['x_positions'])
    o_coord = list(game_status['o_positions'])
    for i in x_coord: # 보드에 합치기
        board_list1[i[0]][i[1]] = 'X'
    for j in o_coord:
        board_list1[j[0]][j[1]] = 'O'
    def three_duplicate_gate(lst):
        counter = Counter(lst)
        for count in counter.values():
            if count >= 3:
                return True
        return False
    
    def fastcheck(list):
        diagonal1 = [(0, 0), (1, 1), (2, 2)]
        diagonal2 = [(2, 0), (1, 1), (0, 2)]
        if sorted(list)==sorted(diagonal1) or sorted(list)==sorted(diagonal2):
            return True
        fastchecklist=[]
        for a0 in list:
            fastchecklist.append(a0[0])
        if three_duplicate_gate(fastchecklist) == True:
            return True
        fastchecklist=[]
        for a1 in list:
            fastchecklist.append(a1[1])
        if three_duplicate_gate(fastchecklist) == True:
            return True
        else:
            return False
        
    if fastcheck(game_status['x_positions'])==True:
        return "X wins"
    if fastcheck(game_status['o_positions'])==True:
        return "O wins"
    elif len(x_positions)+len(o_positions)==9:
        return "tie"
    else:
        return "not decided"


def display(game_status):
    print("\n\n")
    import random##default_list 22x22 matrix
    default_list=[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],\
      [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],\
          [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],\
              [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],\
                  [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],\
                      [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],\
                          [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],\
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],\
                                  [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],\
                                      [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],\
                                          [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    
    O_list=[[1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 2, 2, 0, 0], [1, 0, 2, 0, 0, 2, 0], [1, 0, 2, 0, 0, 2, 0], [1, 0, 0, 2, 2, 0, 0], [1, 0, 0, 0, 0, 0, 0]]
    X_list=[[1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0], [1, 0, 2, 0, 0, 2, 0], [1, 0, 0, 2, 2, 0, 0], [1, 0, 0, 2, 2, 0, 0], [1, 0, 2, 0, 0, 2, 0], [1, 0, 0, 0, 0, 0, 0]]
    
    board_list1=making_board(3,3)
    x_coord = game_status['x_positions']
    o_coord = game_status['o_positions']
    
    for i in x_coord: # 보드에 합치기
        board_list1[i[0]][i[1]] = 'X'
    for j in o_coord:
        board_list1[j[0]][j[1]] = 'O'
    
    pictograph="☂☏☯☮✈✉☕✌❄⻄⛅⚠♻✨∀☘⌚⏩⏭"
    pictograph2="☂❄⛅☀✨"
    randompicto = lambda : str(pictograph[random.randint(0,len(pictograph)-1)])
    randompicto2 = lambda : str(pictograph2[random.randint(0,len(pictograph2)-1)])

    def insert_submatrix(matrix, top_left_x, top_left_y, submatrix):
        submatrix_size = len(submatrix)
        if (0 <= top_left_x <= 22 - submatrix_size) and (0 <= top_left_y <= 22 - submatrix_size):
            for i in range(submatrix_size):
                for j in range(submatrix_size):
                    matrix[top_left_x + i][top_left_y + j] = submatrix[i][j]
        else:
            print("Invalid top-left corner for the submatrix")

    def convert012(status):
        if status == 2:
            return randompicto2()
        elif status == 1:
            return randompicto()
        else:
            return 'ㅤ'
    def print_matrix_convert(matrix):
        for row in matrix:
            print("".join(convert012(status) for status in row))

    for i in range(len(board_list1)):
        for j in range(len(board_list1)):
            if board_list1[i][j]=='X':
                insert_submatrix(default_list, i*7, j*7, X_list)
            if board_list1[i][j]=='O':
                insert_submatrix(default_list, i*7, j*7, O_list)


    print_matrix_convert(default_list)
    print("\n\n")