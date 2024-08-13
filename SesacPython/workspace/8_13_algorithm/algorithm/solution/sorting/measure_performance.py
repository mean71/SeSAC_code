import matplotlib.pyplot as plt 
import os 
import pickle 
import random 
from time import time 

import sorting 

result_dir = 'experiment result'

def generate_testcases(n = 1000):
    testcases = []
    
    for i in range(10, n//10 + 10): 
        testcases.append([random.randint(1, 2*(k+1)) for k in range(10000*(i+1))])

    return testcases

def plot_line_graph(data, save_to = 'sample.png', title="Line Graph", x_label="X-axis", y_label="Y-axis"):

    plt.figure(figsize=(10, 6))
    plt.scatter([e[0] for e in data], [e[1] for e in data])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig(save_to)
    plt.close()

def measure_time(sort_func):
    data = []
    case_generation_time = 0
    sorting_time = 0
    for i in range(1000, 10000, 100):
        try:
            begin = time()
            case = [random.randint(0, i) for j in range(i)]
            end = time()
            case_generation_time += end - begin
            begin = time()
            lst = sort_func(case)
            end = time()
            sorting_time += end - begin 

            print(i, case_generation_time, sorting_time, case_generation_time / (sorting_time+0.0000001))

            data.append((len(case), end - begin))
        except KeyboardInterrupt:
            break 

    print('time segment : ', case_generation_time / (sorting_time+0.0000001))


    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    plot_line_graph(data, save_to = f'{result_dir}/{sort_func.__name__}.png', title = f'{sort_func.__name__} graph', x_label = 'list length', y_label = 'sorting time')

if __name__ == '__main__':
    # begin = time()
    # measure_time(sorted)
    # end = time()
    # print(end - begin)
    # begin = time()
    # measure_time(sorting.sort3_insert)
    # end = time()
    # print(end - begin)
    begin = time()
    measure_time(sorting.merge_sort)
    end = time()
    print(end - begin)
    begin = time()
    measure_time(sorting.quick_sort)
    end = time()
    print(end - begin)