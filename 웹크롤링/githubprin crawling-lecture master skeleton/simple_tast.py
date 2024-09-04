import time

def task(task_number, duration):
    print(f'started {task_number}')
    time.sleep(duration) # 기다
    print(f'ended {task_number}')

def main():
    begin = time.time()
    task1 = (1, 3)
    task2 = (2, 2)
    task3 = (3, 1)
    
    task(*task1)
    task(*task2)
    task(*task3)
    end = time.time()
    
    print(end - begin)

main()