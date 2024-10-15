T = int(input())
for t in range(T):
    N = int(input())
    ai = list(map(int, input().split()))
    min_n, max_n = ai[0], ai[0]
    for x in ai[1:]:
        if x > max_n: max_n = x
        elif x< min_n: min_n = x
    print( f"#{t+1} {max_n - min_n}" )