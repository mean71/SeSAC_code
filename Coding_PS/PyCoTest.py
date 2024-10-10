# s4 1018번 체스판 다시 칠하기
import sys
input = sys.stdin.readline
N,M  = map(int,input().split())
board = [input()for _ in range(N)]
board1 = 'WBWBWBWB','BWBWBWBW'
board2 = 'BWBWBWBW','WBWBWBWB'
min_b = []

for n in range(0,N-7):
    for m in range(0,M-7):
        b1,b2 = 0,0
        for i in range(8):
            for j in range(8):
                temp = board[n+i][m+j]
                b1 += temp!=board1[i%2][j]
                b2 += temp!=board2[i%2][j]
        min_b.append(min(b1,b2))
print(min(min_b))