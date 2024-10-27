import time
import functools
import hashlib
import json
import pickle
from datetime import datetime
import os 

from collections import defaultdict 
from typing import Any, Callable, List, Tuple

def select_by_coverage(
        hist: dict[Any, int], 
        coverage: float = 0.999, 
        key: Callable = lambda x: x[1],
        reverse: bool = True, 
) -> List[Tuple[Any, int]]:
    
    lst: List[Tuple[Any, int]] = list(hist.items())

    lst = sorted(lst, key = key, reverse = reverse)
    total = sum([e[1] for e in lst])
    s = 0

    for idx, (elem, freq) in enumerate(lst): 
        # s += freq 
        if s > total * coverage:
            break 
        s += freq 
    
    return lst[:idx]

def generate_histogram(
        lst: List[Any], 
        key: Callable = lambda x:x, 
        default: Callable = int, 
) -> defaultdict[Any, int]:

    res: defaultdict[Any, int] = defaultdict(default)

    for elem in lst:
        res[key(elem)] += 1 

    return res

def inspect_and_cache(func):
    """
    If the argument func have been executed with inputs (*args, **kargs), DO NOT execute the function again; rather, load the result from cache of the previous run. 

    For each run, print(or write to some file, such as log.txt) the log of the execution, including 
    
    - What function was executed. 
    - When does the execution happen. 
    - How long does the execution take. 
    - What arguments that the execution accepts. 

    ex) If the function make_some_noise(decibel: int) was executed with decibel = 10, log should be like below; 
    =================================
    2024.10.25 13:46 
    function make_some_noise executed with decibel = 10, takes 10.252s
    =================================

    Tip) Since you have to 'update' the log, you should open log file in 'a' mode.
    
    func 인수가 입력(*args, **kargs)으로 실행된 경우 함수를 다시 실행하지 마십시오. 대신 이전 실행의 캐시에서 결과를 로드하세요. 

    각 실행에 대해 다음을 포함하는 실행 로그를 인쇄(또는 log.txt와 같은 일부 파일에 기록)합니다. 
    
    - 어떤 기능이 실행되었는지. 
    - 처형은 언제 이루어지나요? 
    - 실행 시간은 얼마나 걸리나요? 
    - 실행에서 허용되는 인수는 무엇입니까? 

    ex) make_some_noise(decibel: int) 함수를 decibel = 10으로 실행한 경우 로그는 다음과 같아야 합니다. 
    =================================
    2024.10.25 13:46 
    make_some_noise 함수는 데시벨 = 10으로 실행되며 10.252초가 소요됩니다.
    =================================

    Tip) 로그를 '업데이트'해야 하므로 로그 파일을 'a' 모드로 열어야 합니다.
    """
    def f(*args, **kargs):
        return
    return f

if __name__ == '__main__':
    import os 

    from config import DATA_DIR
    
    # testing select_by_coverage 
    hist = {
        'a' : 10,
        'b' : 5,
        'c' : 1,
        'd' : 1,
    }
    print(select_by_coverage(hist, coverage = 0.8))

    # testing generate_histogram
    english_tokens: list[str] = []
    with open(os.path.join(DATA_DIR, 'eng-fra.txt'), 'r', encoding = 'utf-8') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            eng = line[0] 

            for tok in eng.split():
                english_tokens.append(tok) 

    hist: defaultdict[str, int] = generate_histogram(english_tokens)

    lst: List[Tuple[str, int]] = select_by_coverage(hist, coverage = 0.3)
    
    for k, v in lst:
        print(k, v)
    print(len(lst))
    
    length_hist: defaultdict[int, int] = generate_histogram(english_tokens, key = len)

    lst: List[Tuple[int, int]] = select_by_coverage(length_hist, coverage = 0.99)
    
    for k, v in lst:
        print(k, v)
    print(len(lst))

    lst: List[Tuple[int, int]] = select_by_coverage(length_hist, coverage = 0.99, key = lambda x:x[0], reverse = False)
    
    for k, v in lst:
        print(k, v)
    print(len(lst))