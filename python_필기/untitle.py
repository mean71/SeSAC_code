# T = int(input())
# a = []
# for i in range(5):
#     A,B=map(int, input().strip().split())
#     a.append(A+B)
# for i in range(5):
#     print(f'Case #{i+1}: {a[i]}')
# '''
#   5
# 1 1
# 2 3
# 3 4
# 9 8
# 5 2'''
import sys
from io import StringIO
input_data ='''15
push 1
push 2
front
back
size
empty
pop
pop
pop
size
empty
pop
push 3
empty
front
'''
sys.stdin = StringIO(input_data)

def pushX(push,X):
    # if A[0] == 'push'
    pass

N = int(input())
for i in range(N):
    A= input().strip().split()
    print(A)