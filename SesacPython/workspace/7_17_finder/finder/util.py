import math

def mean(lst):                  # 리스트를 받아서  평균(mean)값을 반환
    return sum(lst)/len(lst)

def var(lst):                   # 리스트를 받아서 모든 요소에 mean평균값을 빼고 제곱하여 다시평균을 냄(분산)
    m = mean(lst)
    v = [(e - m)**2 for e in lst]
    return mean(v)
    
    # 이하는 forces 11,12 line 에 호출용
def dist(src, dest):
    return math.sqrt((src[0]-dest[0])**2 + (src[1]-dest[1])**2) #두점사이거리계산공식 (a1-b1)**+(a2-b2)**의 제곱근 실행
    
def dist_unitvec(src, dest):
    d = dist(src, dest) # 두점사이거리계산
    return [(dest[0] - src[0])/d, (dest[1] - src[1])/d] # (dest-src)/거리 : 단위벡터계산