# 4861 회문
def check(word, M):
    for i in range(len(word)+1-M):
        w = word[i: i+M]
        for i in range(len(w)//2):
            if w[i] != w[-i-1]:
                break
        else:return ''.join(w)

for t in range(int(input())):
    N,M = map(int, input().split())
    words = [input() for _ in range(N)]
    for x in words:
        res = check(x, M)
        if res: break
    else:
        for y in list(zip(*words)):
            res = check(y, M)
            if res: break
    print(f'#{t+1} {res}')