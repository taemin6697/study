N,M = map(int,input().split())
visitde = [False] * N
out = []

def solve(depth,index,N,M):
    if depth == M:
        print(' '.join(map(str,out)))
        return
    for i in range(index,N):
        if visitde[i] == False:
            visitde[i] = True
            out.append(i+1)
            solve(depth+1,i,N,M)
            visitde[i] = False
            out.pop()
        else:
           out.append(i+1)
           solve(depth+1,i,N,M)
           visitde[i] = False
           out.pop()
solve(0,0,N,M)
