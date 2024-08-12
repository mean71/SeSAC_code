def get_insert_idx(res, elem, cmp = lambda x, y: x if x > y else y, ):

    for i, e in enumerate(res):
        case = cmp(elem, e)
        if elem == case: # elem > e:
            return i 
    
    return len(res)
# res 배열과 요소를 인수로 받아서 인덱스와 요소를 차례대로 추출해서 받은 lst의 순환중인 elem과 비교 elem이 더크면 res 인덱스 반환

def sort3_insert(lst, cmp = lambda x, y: x if x > y else y):
    res = []

    for elem in lst:
        new_idx = get_insert_idx(res, elem, cmp = cmp) 
        res.insert(new_idx, elem)
    
    return res 
# elem보다 작은 res요소의 첫new_idx를 반환하고, res.insert(new_idx, elem)하여 큰것부터 정렬

# 병합정렬 알고리즘
def merge_sort(lst, cmp = lambda x, y: x if x > y else y):
    if len(lst) <= 1:return lst
    
    merge_sort_lst = []
    mid = len(lst)//2
    L_lst = merge_sort(lst[:mid])
    R_lst = merge_sort(lst[mid:])

    while len(L_lst) > 0 and len(R_lst) > 0:
        if cmp(L_lst[0], R_lst[0]) == L_lst[0]:
            merge_sort_lst.append(R_lst.pop(0))
        else:
            merge_sort_lst.append(L_lst.pop(0))
    merge_sort_lst.extend(L_lst)
    merge_sort_lst.extend(R_lst)
    return merge_sort_lst

def quick_sort(lst, cmp = lambda x, y: x if x > y else y): # start mid end중위원소를 기준으로 삼아

    return lst 

def tim_sort(lst, cmp = lambda x, y: x if x > y else y):

    return lst 


print(merge_sort([1,7,8,3,2,6,9,3,2,5,7,8,4,5,4,5,8,1,4,6,7,9,7,4]))