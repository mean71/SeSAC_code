def get_insert_idx(res, elem, 
        cmp = lambda x, y: x if x > y else y, ):

    for i, e in enumerate(res):
        case = cmp(elem, e)
        if elem == cmp(elem, e): # elem > e:
            return i 
    
    return len(res)


def sort3_insert(lst, cmp = lambda x, y: x if x > y else y):
    res = []

    for elem in lst:
        new_idx = get_insert_idx(res, elem, cmp = cmp)
        res.insert(new_idx, elem)
    
    return res 


def merge_sort(lst, cmp = lambda x, y: x if x > y else y):
    return lst 

def quick_sort(lst, cmp = lambda x, y: x if x > y else y):
    return lst 

def tim_sort(lst, cmp = lambda x, y: x if x > y else y):
    return lst 