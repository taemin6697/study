N, M = map(int, input().split())
visited = [False] * N  # 탐사 여부 check
out = []  #1,
k = 0
def solve(depth, N, M):
    if depth == M:  # 탈출 조건
        print(' '.join(map(str, out)),depth)# list를 str으로 합쳐 출력
        return #리턴을 하면 전 상태로 돌아옴 즉 depth = 1
    for i in range(N):  # 탐사 check 하면서
        if visited[i]==False:  # 탐사 안했다면
            visited[i] = True  # 탐사 시작(중복 제거)
            out.append(i+1)  # 탐사 내용
            print(i,visited,out,depth)
            solve(depth+1, N, M)  # 깊이 우선 탐색
            print(depth)
            visited[i] = False  # 깊이 탐사 완료
            out.pop()  # 탐사 내용 제거
            print(i,visited,out,depth)


solve(0, N, M)
