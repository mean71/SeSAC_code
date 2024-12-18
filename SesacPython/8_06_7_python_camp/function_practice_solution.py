import random 

# --------------------------------------------
# 1. max / min 구현하기 
#
# cmp라는 함수를 이용한 min/max 구현하기. 
# cmp는 두 원소 중 더 큰 것을 반환하는 함수. 
# --------------------------------------------


def my_max(lst, cmp = lambda x, y: x):
    pass 

def my_min(lst, cmp = lambda x, y: x):
    pass 

lst = [(1,2), (2,3), (5,3), (19, 2), (6, 100)]

def max_by_first_element(lst):
    max_elem = lst[0]

    for e in lst[1:]:
        if e[0] > max_elem[0] :
            max_elem = e 
    
    return max_elem 

# print(max_by_first_element(lst))

def max_by_sum(lst):
    max_elem = lst[0]

    for e in lst[1:]:
        if sum(e) > sum(max_elem):
            max_elem = e 
    
    return max_elem 

def my_max(lst, cmp = lambda x,y:x):
    max_elem = lst[0]

    for e in lst[1:]:
        max_elem = cmp(e, max_elem)
    
    return max_elem 

def compare_by_sum(x, y):
    if sum(x) > sum(y): return x
    else: return y 

def compare_by_first_element(x, y):
    if x[0] > y[0]: return x 
    else: return y 

# print(my_max(lst, cmp = compare_by_sum))
# print(my_max(lst, cmp = compare_by_first_element))

def my_min(lst, cmp = lambda x, y:x, tie_breaker = lambda x,y:x):
    min_elem = lst[0]

    for e in lst[1:]:
        case = cmp(e, min_elem)
        if case == min_elem:
            min_elem = e
        elif case == e:
            pass 
        else:
            tie_case = tie_breaker(e, min_elem)
            if tie_case == min_elem:
                min_elem = e 

    return min_elem

def my_compare(x, y):
    if x[0] > y[0]: return x
    elif y[0] > x[0]: return y
    else: return 'equal'


# print(my_min([(1,2,), (1,3)], cmp = my_compare, tie_breaker = lambda x, y: x if x[1] > y[1] else y))
# print(max_by_sum(lst))

# --------------------------------------------
# 2. sort 구현하기 
# 
# 1) 그냥 순서대로 오름차순으로 정렬하기 
# 2) 오름차순, 내림차순으로 정렬하기 
# 3) 주어진 기준 cmp에 맞춰서 오름차순, 내림차순으로 정렬하기 
# 4) 주어진 기준 cmp가 큰 element를 출력하거나, 같다는 결과를 출력하게 만들기 
# 5) cmp상 같은 경우 tie-breaking하는 함수 넣기 
# --------------------------------------------

def sort1_min(lst):
    res = []
    c = [e for e in lst]
    n = len(lst)
    
    while len(res) < n:
        m = my_min(c, lambda x, y: x if x > y else y) 
        res.append(m) 
        c.remove(m) 
        
    return res

def get_insert_idx(res, elem, 
        upper_to_lower = False, 
        cmp = lambda x, y: x if x>y else y, 
        tie_breaker = lambda x, y: x):
    
    tie_flag = False
    for i, e in enumerate(res):
        case = cmp(elem, e)
        if case != elem and case != e:
            tie_flag = True
        
        if tie_flag:
            tie_case = tie_breaker(elem, e)
            if not upper_to_lower and tie_case == e:
                return i 
            elif upper_to_lower and elem == tie_case:
                return i 
        elif not upper_to_lower:
            if e == case: # elem < e:
                return i
        else:
            if elem == cmp(elem, e): # elem > e:
                return i 
    
    return len(res)

# print(get_insert_idx([(0, 1), (1,1,), (1,3), (1,10), (1,15), (2,5)], (1,2), upper_to_lower = False, cmp = my_compare, tie_breaker = lambda x, y: x if x[1] > y[1] else y))

# print(get_insert_idx([1,3,5], 2, upper_to_lower = False)) # 1
# print(get_insert_idx([6,3,1], 2, upper_to_lower = True)) #  2
# print(get_insert_idx(
#     [(1,3), (3,56), (12,1)], (10,1), upper_to_lower = False, cmp = lambda x,y:x if x[0] > y[0] else y))


def sort1_insert(lst):
    res = []

    for idx, elem in enumerate(lst):
        new_idx = get_insert_idx(res, elem)
        res.insert(new_idx, elem)
    
    return res 

lst = [10, 1, 6, 4, 2, 5,]
# print(sort1_min(lst))
# print(sort1_insert(lst))

def sort2(lst, upper_to_lower = True):
    pass 

def sort2_min(lst, upper_to_lower = True):
    res = []
    c = [e for e in lst]
    n = len(lst)
    
    while len(res) < n:
        m = my_min(c, lambda x, y: x if x > y else y) 
        if not upper_to_lower:
            res.append(m) 
        else:
            res = [m] + res
        c.remove(m) 
        
    return res

def sort2_insert(lst, upper_to_lower = True):
    res = []

    for elem in lst:
        new_idx = get_insert_idx(res, elem, upper_to_lower = upper_to_lower)
        res.insert(new_idx, elem)
    
    return res 

# print(sort2_min(lst, upper_to_lower = True))
# print(sort2_insert(lst, upper_to_lower = True))
# print(sort2_min(lst, upper_to_lower = False))
# print(sort2_insert(lst, upper_to_lower = False))

def sort3(lst, upper_to_lower = True, cmp = lambda x, y: x):
    pass 

def sort3_min(lst, upper_to_lower = True, cmp = lambda x,y:x):
    res = []
    c = [e for e in lst]
    n = len(lst)
    
    while len(res) < n:
        m = my_min(c, cmp) 
        if not upper_to_lower:
            res.append(m) 
        else:
            res = [m] + res
        c.remove(m) 
        
    return res

def sort3_insert(lst, upper_to_lower = True, cmp = lambda x,y:x):
    res = []

    for elem in lst:
        new_idx = get_insert_idx(res, elem, upper_to_lower = upper_to_lower, cmp = cmp)
        res.insert(new_idx, elem)
    
    return res 

# print(sort3_insert(lst, cmp = lambda x,y: x if x>y else y))
# print(sort3_insert(lst, cmp = lambda x,y: x if x<y else y))

# tuple_lst = [ (1,3),(1,2), (2,3), (5,3), (19, 2), (6, 100)]
# print(sort3_insert(tuple_lst, cmp = lambda x,y: x if x[0]>y[0] else y))
# print(sort3_insert(tuple_lst, cmp = lambda x,y: x if sum(x)>sum(y) else y))


def sort4(lst, upper_to_lower = True, cmp = lambda x, y: x):
    pass 

def sort5(lst, upper_to_lower = True, cmp = lambda x, y: x, tie_breaker = lambda x, y: random.choice([x,y])):
    pass 

def sort5_min(lst, upper_to_lower = True, cmp = lambda x, y: x, tie_breaker = lambda x, y: random.choice([x,y])):
    res = []
    c = [e for e in lst]
    n = len(lst)
    
    while len(res) < n:
        m = my_min(c, cmp, tie_breaker = tie_breaker) 
        if not upper_to_lower:
            res.append(m) 
        else:
            res = [m] + res
        c.remove(m) 
        
    return res


def sort5_insert(lst, upper_to_lower = True, cmp = lambda x, y: x, tie_breaker = lambda x, y: random.choice([x,y])):
    res = []

    for elem in lst:
        new_idx = get_insert_idx(res, elem, upper_to_lower = upper_to_lower, cmp = cmp, tie_breaker = tie_breaker)
        res.insert(new_idx, elem)
    
    return res 


tuple_lst = [ (1,3),(1,2), (1,5), (1,4),(2,3), (5,3), (19, 2), (6, 100)]
print(sort5_insert(tuple_lst, cmp = my_compare, tie_breaker = lambda x, y: x if x[1] > y[1] else y))


# --------------------------------------------
# os_file_concept.py 해보고 올 것 
# --------------------------------------------

# --------------------------------------------
# 3. safe pickle load/dump 만들기 
# 
# 일반적으로 pickle.load를 하면 무조건 파일을 읽어와야 하고, dump는 써야하는데 반대로 하면 굉장히 피곤해진다. 
# 이런 부분에서 pickle.load와 pickle.dump를 대체하는 함수 safe_load, safe_dump를 짜 볼 것.  
# hint. 말만 어렵고 문제 자체는 정말 쉬운 함수임.
# --------------------------------------------

def safe_load(pickle_path):
    pass 

def safe_dump(pickle_path):
    pass 


# --------------------------------------------
# 4. 합성함수 (추후 decorator)
# 
# 1) 만약 result.txt 파일이 없다면, 함수의 리턴값을 result.txt 파일에 출력하고, 만약 있다면 파일 내용을 읽어서 
#    '함수를 실행하지 않고' 리턴하게 하는 함수 cache_to_txt를 만들 것. txt 파일은 pickle_cache 폴더 밑에 만들 것.  
# 2) 함수의 실행값을 input에 따라 pickle에 저장하고, 있다면 pickle.load를 통해 읽어오고 없다면 
#    실행 후 pickle.dump를 통해 저장하게 하는 함수 cache_to_pickle을 만들 것. pickle 파일은 pickle_cache 폴더 밑에 만들 것. 
# 3) 함수의 실행값을 함수의 이름과 input에 따라 pickle에 저장하고, 2)와 비슷하게 진행할 것. pickle 파일은 pickle_cache 폴더 밑에, 각 함수의 이름을 따서 만들 것 
# --------------------------------------------

def cache_to_txt(function):
    pass 

def cache_to_pickle(function):
    
    def f(x):
        pickle_path = 'pickle_path'
        file_path = f'{pickle_path}/{x}.pickle'
        
        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        
        if not os.path.exists(file_path):
            res = function(x)
            pickle.dump(res, open(file_path, 'wb+'))
        else:
            res = pickle.load(open(file_path, 'rb'))
        
        return res  
    
    return f 

def double(x):
    return 2*x

# print(cache_to_pickle(double)(2))

# stdout 
# >> 4 
# file 
# pickle_cache/2.pickle

if __name__ == '__main__':
    pass