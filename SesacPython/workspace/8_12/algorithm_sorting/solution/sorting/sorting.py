
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
    if len(lst) > 1:
        mid = len(lst) // 2  
        l = lst[:mid]  
        r = lst[mid:]

        return merge(merge_sort(l), merge_sort(r), lst)
    else:
        return lst


def merge(l, r, lst, cmp = lambda x, y: x if x > y else y):
    i, j, k = 0, 0, 0
    
    while i < len(l) and j < len(r):
        if cmp(l[i], r[j]) == r[j]:
            lst[k] = l[i]
            i += 1
        else:
            lst[k] = r[j]
            j += 1
        k += 1

    # Checking if any element was left
    while i < len(l):
        lst[k] = l[i]
        i += 1
        k += 1

    while j < len(r):
        lst[k] = r[j]
        j += 1
        k += 1
    
    return lst 


def partition(lst, low, high, cmp = lambda x, y: x if x > y else y):
    i = low - 1  # index of smaller element
    pivot = lst[high]  # pivot

    for j in range(low, high):
        # If current element is smaller than or equal to pivot
        if cmp(lst[j], pivot) == pivot:
            i = i + 1
            lst[i], lst[j] = lst[j], lst[i]  # swap

    lst[i + 1], lst[high] = lst[high], lst[i + 1]  # swap
    return i + 1


def quick_sort(lst, cmp = lambda x, y: x if x > y else y):

    return quick_sort_util(lst, 0, len(lst)-1, cmp = lambda x, y: x if x > y else y)


def quick_sort_util(lst, low, high, cmp = lambda x, y: x if x > y else y):
    if low < high:
        partition_index = partition(lst, low, high, cmp = lambda x, y: x if x > y else y)

        # recursively sort elements before partition and after partition
        quick_sort_util(lst, low, partition_index - 1)
        quick_sort_util(lst, partition_index + 1, high)

    return lst 

