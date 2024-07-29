def manual_finder(f):
    while True:
        i = input(f'Guess the argument!\nGuess is: ')
        res = f(float(i))
        if res is True:
            print(f'You found the right argument!; {float(i)}')
            return float(i)
        print(res)
        
def naive_finder(f, lst = list(range(10000001))):
    for i in lst:
        res = f(float(i))
        if res is True:
            print(f'You found the right argument!; {float(i)}')
            return float(i)
        print(res)
        
def smart_finder(f, low = 0, high = 10000000, precision=0.001):
    while low <= high:
        mid = (low + high) / 2
        res = f(mid)
        if res == 'down':
            high = mid - precision
        elif res == 'up':
            low = mid + precision
        elif res is True:
            print(f'You found the right argument!; {mid}')
            return mid
    print('No valid argument found within range.')
    return None