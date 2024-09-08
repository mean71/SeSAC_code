import sys
from io import StringIO

# input_data ='''3
# 3 7
# 15 7
# 5 2'''
# sys.stdin = StringIO(input_data)

# 2563 색종이 s5
n, extent = int(sys.stdin.readline()), []
for _ in range(n):
    a,b = map(int, sys.stdin.readline().split())
    extent += [ (a+i+1,b+j+1) for i in range(10) for j in range(10)]
print(len(set(extent)))