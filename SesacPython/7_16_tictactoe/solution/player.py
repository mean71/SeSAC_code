from random import randint 

def random_player(x_or_o, x_positions, o_positions):
    move = (0, 0)
    while move in x_positions + o_positions:
        x = randint(0, 2)
        y = randint(0, 2)
        move = (x, y)
    return move 

def smart_player(x_or_o, x_positions, o_positons):
    
    def winning_move(positions, opponent_positions):
        # Check if there is a move that can win the game
        for row in range(3):
            for col in range(3):
                if (row, col) not in positions and (row, col) not in opponent_positions:
                    new_positions = positions + [(row, col)]
                    if is_winning(new_positions):
                        return (row, col)
        return None

    def is_winning(positions):
        # Check if a set of positions is a winning set
        winning_combinations = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]
        ]
        for combination in winning_combinations:
            if all(pos in positions for pos in combination):
                return True
        return False

    def opponent(x_or_o):
        return 'X' if x_or_o == 'O' else 'O'

    player_positions = x_positions if x_or_o == 'X' else o_positions
    opponent_positions = o_positions if x_or_o == 'X' else x_positions

    # Step 1: Win if possible
    move = winning_move(player_positions, opponent_positions)
    if move:
        return move

    # Step 2: Block opponent's winning move
    move = winning_move(opponent_positions, player_positions)
    if move:
        return move

    # Step 3: Take the center if available
    if (1, 1) not in player_positions and (1, 1) not in opponent_positions:
        return (1, 1)

    # Step 4: Take a corner if available
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    for corner in corners:
        if corner not in player_positions and corner not in opponent_positions:
            return corner

    # Step 5: Take any remaining side
    sides = [(0, 1), (1, 0), (1, 2), (2, 1)]
    for side in sides:
        if side not in player_positions and side not in opponent_positions:
            return side

    # Step 6: Take any remaining spot
    for row in range(3):
        for col in range(3):
            if (row, col) not in player_positions and (row, col) not in opponent_positions:
                return (row, col)

    return None  # This should never be reached if the game is played correctly
