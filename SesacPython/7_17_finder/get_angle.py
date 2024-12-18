from animation import check_collision   #
from finder import naive_finder         #호출은 했는데 없음 채워야 하나

def shoot():
    angle = 0

    for angle in range(3,6):
        print(angle*10)
        if check_collision(angle*10) is True:
            return angle 
    
    return 'Failed to find angle'

if __name__ == '__main__':
    print(shoot())