T = int(input())
a = []
for i in range(5):
    A,B=map(int, input().strip().split())
    a.append(A+B)
for i in range(5):
    print(f'Case #{i+1}: {a[i]}')
'''
  5
1 1
2 3
3 4
9 8
5 2'''