def manual_finder(f):
    while True:
        i = input(f'Guess the argument!\nGuess is: ')
        res = f(float(i))
        if res is True:
            print(f'You found the right argument!; {float(i)}')
            return float(i)
        print(res) 
def naive_finder(f, lst = list(range(5))):
    binary = len(lst) / 2
    answer = lst[int(binary)]
    while True :
        res = f(float(answer))
        if res == 'up' :
            binary = binary + abs(binary / 2)
            answer = lst[int(binary)]
            res = f(float(answer))
        elif res == 'down' :
            binary = binary - abs(binary / 2)
            answer = lst[int(binary)]
            res = f(float(answer))
        if res is True:
            print(f'You found the right argument!; {float(answer)}')
            return float(answer)
def smart_finder(f, min_input = 0, max_input = 100):
    answer = (min_input + max_input) / 2
    while True :
        res = f(float(answer))
        if res == 'up' :
            answer = answer + answer / 2
            res = f(float(answer))
        elif res == 'down' :
            answer = answer - answer / 2
            res = f(float(answer))
        elif res == None :
            answer = answer + answer / 2
            res = f(float(answer))
        if res is True:
                print(f'You found the right argument!; {float(answer)}')
                return float(answer)