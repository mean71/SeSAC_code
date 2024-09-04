import updown_game
def manual_finder(f):
    while True:
        i = input(f'Guess the argument!\nGuess is: ')
        res = f(float(i))
        if res is True:
            print(f'You found the right argument!; {float(i)}')
            return float(i)           
        print(res) 
    

def naive_finder(f, lst = list(range(5))):
    import time
    import sys
    loading_symbols = ['|', '/', '-', '∖']
    guess_naive_up=0
    guess_naive_down=10000000
    guess_naive=10000000//2
    idx = 0
    while True:        
        sys.stdout.write(f'\r {guess_naive} Guessing... {loading_symbols[idx]}')
        idx = (idx + 1) % len(loading_symbols)  
        time.sleep(0.08)
        sys.stdout.flush()
        result_guess = f(guess_naive)   
        if result_guess=='up':
            guess_naive_up=guess_naive
            guess_naive=(guess_naive_down+guess_naive_up)//2              
        if result_guess=='down':    
            guess_naive_down=guess_naive
            guess_naive=(guess_naive_down+guess_naive_up)//2   
        if result_guess==True:
            print("\n===found arguement===")
            return guess_naive
def smart_finder(f, min_input = 0, max_input = 100):    
    import time
    import sys
    loading_symbols = ['|', '/', '-', '∖']  
    guess_smart_up=min_input
    guess_smart_down=max_input
    guess_smart=max_input/2
    idx = 0
    while True:
        sys.stdout.write(f'\r {guess_smart} Guessing... {loading_symbols[idx]}')
        idx = (idx + 1) % len(loading_symbols)
        time.sleep(0.08)
        result_guess = f(guess_smart)    
        if result_guess=='up':
            guess_smart_up=guess_smart
            guess_smart=(guess_smart_down+guess_smart_up)/2              
        if result_guess=='down':
            guess_smart_down=guess_smart
            guess_smart=(guess_smart_down+guess_smart_up)/2   
        if result_guess==True:
            print("\n===found arguement===")
            return guess_smart