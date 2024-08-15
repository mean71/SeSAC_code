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
input_data ='''ABCDE
abcde
01234
FGHIJ
fghij
'''
sys.stdin = StringIO(input_data) # 입력을 받는대신 input_data에서 테스트케이스를 읽어온다






# import sys

# N = int(input())
# Q = []
# for i in range(N):
#     A = sys.stdin.readline().split()

#     if A[0] == 'pop':
#         if len(Q) != 0: print(Q.pop(0))
#         else: print('-1')

#     elif A[0] == 'size': print(len(Q))

#     elif A[0] == 'empty':
#         if len(Q) == 0: print('1')
#         else: print('0')

#     elif A[0] == 'front':
#         if len(Q) != 0: print(Q[0])
#         else: print('-1')

#     elif A[0] == 'back':
#         if len(Q) != 0: print(Q[-1])
#         else: print('-1')
    
#     else: Q.append(A[1])

    
# push X: 정수 X를 큐에 넣는 연산이다
# pop: 큐에서 가장 앞에 있는 정수를 빼고, 그 수를 출력한다. 만약 큐에 들어있는 정수가 없는 경우에는 -1을 출력한다
# size: 큐에 들어있는  정수의 개수를 출력한다
# empty: 큐가 비어있으면 1, 아니면 0을 출력한다
# front: 큐의 가장 앞에 있는 정수를 출력한다. 만약 큐에 들어있는 정수가 없는 경우에는 -1을 출력한다
# back: 큐의 가장 뒤에 있는 정수를 출력한다. 만약 큐에 들어있는 정수가 없는 경우에는 -1을 출력한다

# import sys

# N = int(input())
# Q = []
# for i in range(N):
#     A = sys.stdin.readline().split()

#     if A[0] == 'pop':
#         if len(Q) != 0: print(Q.pop(0))
#         else: print('-1')

#     elif A[0] == 'size': print(len(Q))

#     elif A[0] == 'empty':
#         if len(Q) == 0: print('1')
#         else: print('0')

#     elif A[0] == 'front':
#         if len(Q) != 0: print(Q[0])
#         else: print('-1')

#     elif A[0] == 'back':
#         if len(Q) != 0: print(Q[-1])
#         else: print('-1')
    
#     else: Q.append(A[1])
# push X: 정수 X를 스택에 넣는 연산이다.
# pop: 스택에서 가장 위에 있는 정수를 빼고, 그 수를 출력한다. 만약 스택에 들어있는 정수가 없는 경우에는 -1을 출력한다.
# size: 스택에 들어있는 정수의 개수를 출력한다.
# empty: 스택이 비어있으면 1, 아니면 0을 출력한다.
# top: 스택의 가장 위에 있는 정수를 출력한다. 만약 스택에 들어있는 정수가 없는 경우에는 -1을 출력한다.


# 백준10798 세로읽기

string = [(sys.stdin.readline().strip()) for _ in range(5)]
strs = ''
print(string,sep='\n')
L = max(map(len, string))
# max_length = max(len(s) for s in strings)
print(L)

for i in range(L):
    for j in range(5):
        if i < len(string[j]):
            strs += string[j][i]
print(strs)