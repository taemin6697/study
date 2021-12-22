N,M = map(int,input().split())
visitde = [False] * N
out = []

def solve(depth,N,M):
    if depth == M:
        print(' '.join(map(str,out)))
        return
    for i in range(N):
        if visitde[i] == False:
            visitde[i] = True
            out.append(i+1)
            solve(depth+1,N,M)
            visitde[i] = False
            out.pop()
solve(0,N,M)
