import torch 
from typing import Any, List

def product_sum(
    l: List[float], 
    r: List[float], 
) -> float:
    return sum([a*b for a, b in zip(l, r)])

def conv1d_list(lst: List[float], 
    kernel: List[float], 
    padding: int = 0, 
    stride: int = 1, 
) -> List[float]:
    # lst = [e_1, e_2, ..., e_n] 
    # kernel = [k_1, k_2, ..., k_m] 
    n = len(lst) + 2 * padding 
    lst = [0] * padding + lst + [0] * padding
    k = len(kernel)
    assert k < n 
    features = []
    
    for i in range(0, n-k+1, stride):
        features.append(product_sum(lst[i:i+k], kernel))

    return features 

def product_loop(*loops):
    if len(loops) == 1:
        return [[e] for e in loops[0]]
    res = []
    for i in loops[0]:
        for rest in product_loop(*loops[1:]):
            res.append([i] + rest)
    return res 
                #do something with i_0, i_1, ... , i_n
def get_dimension(lst):
    if not isinstance(lst[0], list):
        return 1 
    else:
        return 1 + get_dimension(lst[0])

def change_element(lst, indices, elem):
    n = get_dimension(lst)
    assert n == len(indices), (n, indices, lst)
    
    if n == 1:
        lst[indices[0]] = elem 
    else:
        for idx, sub_list in enumerate(lst):
            if idx == indices[0]:
                change_element(sub_list, indices[1:], elem)

def segment_lst(lst, indices):
    res = lst
    for (start, end) in indices:
        res = res[start:end]
    return res 

def get_lst(lst, indices):
    res = lst 
    for i in indices:
        res = lst[i] 
    return res 

def product_sum_ndim(a, b):
    n = get_dimension(a)
    m = get_dimension(b) 

    assert n == m 
    l = len(a)
    s = 0 
    for indices in product_loop(*[range(l) for _ in range(n)]):
        s += get_lst(a, indices) * get_lst(b, indices)
    return s 

def convnd_list(
    lst: Any, 
    kernel: Any, 
    padding: int = 0, 
    stride: int = 1, 
) -> Any:
    N = get_dimension(lst)
    n = len(lst) # element개수
    k = len(kernel)
        """
    n-k+1, n-k+1, ..., n-k+1 (N개)
    [0, 0] ~ [k, k]
    [0, 1] ~ [k, k+1]
    ...
    [n-k, n-k] ~ [n-1, n-1]
    """
    """# for 루프를 N개를 돌아야하는데
    for i_1 in range(0, n-k+1, stride):
        for i_2 in range(0, n-k+1, stride):
            ...
                features[i_1][i_2]...[i_n] = product_sum_ndim(kernel, lst[i_1:i_1+k][i_2:i_2+k]...[i_n:i_n+k])
    """
    features = torch.zeros(*[n for _ in range(N)]).tolist()

    for indices in product_loop([range(0, n-k+1, stride) for _ in range(N)]):
        semgment_range = [(i, i+k) for i in indices]
        segment = segment_lst(lst, semgment_range)
        change_element(features, indices, product_sum_ndim(kernel, segment))
        
    return features 

if __name__ == '__main__':
    import torch 

    lst = conv1d_list(
        list(range(1, 6)), 
        [1, 0, 1]
    )
    # [1, 2, 3, 4, 5] 
    #       [1, 0, 1] 
    # [4, 6, 8]
    print(lst)

    print(product_loop(
        range(0, 5), 
        range(1, 6), 
        range(2, 7), 
    ))
    
    """
    for i in range(0, 5):
        for j in range(1, 6):
            for k in range(2, 7):
                print(i, j, k)
    """
    lst = torch.randn(3,3,3,3).tolist()
    print(get_dimension(lst))

    change_element(lst, [0, 0, 0, 0], 1)
    print(lst)
