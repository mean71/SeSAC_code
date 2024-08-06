# --------------------------------------------
# 1. 패턴 찍는 함수들 만들어보기
# 1) 피라미드 찍어보기 - 1
# 다음과 같은 패턴의 높이를 받아, 다음 패턴을 프린트하는 함수 pyramid1를 짜 보세요.
#     *
#    ***
#   *****
#  *******
# *********
# --------------------------------------------
N=int(input())
def pyramid1(N):
    for n in range(N):
        print(f'{' '*(N-n)}{"*"*(2*n+1)}')
    pass
pyramid1(N)
# write your code here 

# --------------------------------------------
# 2) 피라미드 찍어보기 - 2
# 다음과 같은 패턴의 높이를 받아, 다음 패턴을 프린트하는 함수 pyramid2를 짜 보세요. 
# 
#     * 
#    * * 
#   * * * 
#  * * * * 
# * * * * * 
# --------------------------------------------
def pyramid2(n):
    for n in range(1,N+1):
        print( ' '*(N-n), end='' )
        print( '* '*n)
    pass
N=int(input())
pyramid2(N)
# write your code here 

# --------------------------------------------
# 3) 피라미드 찍어보기 - 3
# 다음 패턴의 높이를 받아, 다음 패턴을 프린트하는 함수 pyramid3를 짜 보세요. 
#     A 
#    A B 
#   A B C 
#  A B C D 
# A B C D E 
# --------------------------------------------
def pyramid3(n):
    x="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for n in range(1, N+1):
        print( ' '*(N-n), end='' )
        for i in range(n):
            if i > 25:
                print( f'{x[i-26]} ', end = '' ) 
            else:
                print( f'{x[i]} ', end = '' )
        print()
    pass
N = int(input())
pyramid3(N)
# write your code here

# --------------------------------------------
# 4) 피라미드 찍어보기 - 4
# 다음 패턴의 높이를 받아, 다음 패턴을 프린트하는 함수 pyramid4를 짜 보세요.
#       1 
#      1 1 
#     1 2 1 
#    1 3 3 1 
#   1 4 6 4 1
#  1 5 10 10 5 1
# --------------------------------------------
def pyramid4(n):

    pass
# write your code here

# --------------------------------------------
# 5) 다음 패턴을 찍는 함수 sierpinski_triangle을 짜 보세요.
# n = 2
#         *
#        * *
#       *   *
#      * * * *
#     *       * 
#    * *     * * 
#   *   *   *   * 
#  * * * * * * * * 
# n = 3 
#                 *
#                * *
#               *   *
#              * * * *
#             *       * 
#            * *     * *
#           *   *   *   * 
#          * * * * * * * *
#         *               *   
#        * *             * *  
#       *   *           *   * 
#      * * * *         * * * * 
#     *       *       *       * 
#    * *     * *     * *     * * 
#   *   *   *   *   *   *   *   * 
#  * * * * * * * * * * * * * * * *
# --------------------------------------------
def sierpinski_triangle(n):

    pass
# write your code here

# 5-2)
def sierpinski_triangle(n):

    pass

# --------------------------------------------
# 2. 여러 리스트 관련 함수들 구현해보기
# 아래 함수들은 대부분 itertools에 있는 함수들임. 
# itertools를 쓰지 말고 구현해 볼 것.  
# 1) accumulate(lst, function = lambda x, y : x+y)
# lst의 각 원소들에 대해서, function을 누적하여 적용한 리스트를 반환. 
# lst -> [lst[0], f(lst[0], lst[1]), f(lst[2], f(lst[1], lst[0])), ...] 
# --------------------------------------------
def accumulate(lst, function = lambda x, y: x+y):
    nlst=[lst[0]]
    for i in range(len(lst)-1):
      nlst.append(function(lst[i], lst[i+1]))
    return nlst
lst=[1,2,3,4,5,6,7,8,9,10]
print(accumulate(lst))
accumulate(lst)
# write your code here

# --------------------------------------------
# 2) batched(lst, n)
# lst의 원소들을 n개의 인접한 원소들끼리 묶은 리스트를 반환. 
# ex) batched([1,2,3,4,5], 2) 
#     >> [(1,2), (3,4), (5,)]
# ex) batched(['a', 'b', 1, 3, 6, 1, 3, 7], 3) 
#     >> [('a', 'b', 1), (3, 6, 1), (1, 3, 7)]
# --------------------------------------------
def batched(lst, n):

    pass
# write your code here

# --------------------------------------------
# 3) product(args)
# list들의 list args를 받아서, 각각의 리스트에서 하나씩의 원소를 뽑은 튜플들의 리스트를 반환. 
# ex) product([[1,2,3], [4,5,6])
#     >> [(1,4), (1,5), (1,6), 
#         (2,4), (2,5), (2,6), 
#         (3,4), (3,5), (3,6),] 
# --------------------------------------------
def product(args):

    pass
# write your code here

# --------------------------------------------
# 4) permutations(lst, r) 
# lst 안의 원소들 r개로 이루어진 permutation의 리스트를 반환. 
# permutation이란, 순서를 가지면서 중복을 허용하지 않는 부분집합을 의미함. 
# 즉 여기서는 1,2와 2,1은 다르고, 1,1은 허용되지 않음. 
# ex) permutations([1,2,3,4,5], 2)
#     >> [(1,2), (1,3), (1,4), (1,5), 
#         (2,1), (2,3), (2,4), (2,5), 
#         (3,1), (3,2), (3,4), (3,5), 
#         (4,1), (4,2), (4,3), (4,5), 
#         (5,1), (5,2), (5,3), (5,4),]
# --------------------------------------------
def permutations(lst, r):

    pass
# write your code here

# --------------------------------------------
# 5) combination(lst, r) 
# lst 안의 원소들 r개로 이루어진 combination의 리스트를 반환. 
# combination이란, 순서를 가지지 않으면서 중복을 허용하지 않는 부분집합을 의미함. 
# 즉 여기서는 1,2와 2,1은 같고, 1,1은 허용되지 않음. 
# ex) combination([1,2,3,4,5], 2)
#     >> [(1,2), (1,3), (1,4), (1,5), 
#         (2,3), (2,4), (2,5), 
#         (3,4), (3,5), 
#         (4,5), ]
# --------------------------------------------
def combination(lst, r):

    pass
# write your code here

# --------------------------------------------
# 6) combination_with_duplicate(lst, r)
# lst 안의 원소들 r개로 이루어진 중복을 허용하는 combination의 리스트를 반환. 
# ex) combination_with_duplicate([1,2,3,4,5], 2)
#     >> [(1,1), (1,2), (1,3), (1,4), (1,5), 
#         (2,2), (2,3), (2,4), (2,5), 
#         (3,3), (3,4), (3,5), 
#         (4,4), (4,5),
#         (5,5), ]
# --------------------------------------------
def combination_with_duplicate(lst, r):

    pass
# write your code here