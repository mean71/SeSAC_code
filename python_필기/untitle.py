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
input_data ='''9
4
1 3
1 5
3
2
5
2
2
5
'''
sys.stdin = StringIO(input_data) # 입력을 받는대신 input_data에서 테스트케이스를 읽어온다


# 백준 28278
import sys
class Node:
    def __init__(self, data, next = None):
        self.data = data
        self.next = next
class Link:
    def __init__(self):
        self.head = None
        self.end = None
        self.size = 0
    
    def one(self, elem): # 정수n[1]를 스택에 넣는다
        node = Node(elem)
        self.size += 1
        if self.head is None:
            self.head = node
            self.end = self.head
        else:
            self.end.next = node
            self.end = node
            
    def two(self): # 스택에 정수가 있다면 맨 위 정수 빼고 출력한다. 없다면 -1
        if self.head is None: return -1
        elif self.head == self.end:
            res = self.end.data
            self.head = None
            self.end = None
            self.size -= 1
            return res
        else:
            res = self.end.data
            cur = self.head
            self.size -= 1
            while cur.next != self.end:
                cur = cur.next
            cur.next = None
            self.end = cur
            self.size -= 1
            return res
    def three(self): # 스택에 들어있는 정수 개수 출력
        return self.size
    def four(self): # 스택이 비어있으면 1, 아니면 0 출력
        if self.size == 0: return 1
        else: return 0
    def five(self): # 스택에 정수가 있다면 맨위 정수 출력한다. 없다면 -1
        if self.head is None: return -1
        else: return self.end.data
        
def stack():
    N = int(sys.stdin.readline())
    link = Link()
    for i in range(N):
        n = list(map(int, sys.stdin.readline().strip().split()))
        if n[0] == 1:# 정수n[1]를 스택에 넣는다
            link.one(n[1])
        elif n[0] == 2:# 스택에 정수가 있다면 맨 위 정수 빼고 출력한다. 없다면 -1
            print(link.two())
        elif n[0] == 3:# 스택에 들어있는 정수 개수 출력
            print(link.three())
        elif n[0] == 4:# 스택이 비어있으면 1, 아니면 0 출력
            print(link.four())
        elif n[0] == 5:# 스택에 정수가 있다면 맨위 정수 출력한다. 없다면 -1  
            print(link.five())
stack()



# import sys

# N = int(sys.stdin.readline())
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

# string = [(sys.stdin.readline().strip()) for _ in range(5)]
# strs = ''
# print(string,sep='\n')
# L = max(map(len, string))
# # max_length = max(len(s) for s in strings)
# print(L)

# for i in range(L):
#     for j in range(5):
#         if i < len(string[j]):
#             strs += string[j][i]
# print(strs)