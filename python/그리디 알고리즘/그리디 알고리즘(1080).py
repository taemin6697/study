def change(x,y,A):
    for i in range(3):
        for j in range(3):
            if A[x+i][y+j] == 0:
                A[x+i][y+j] = 1
            else:
                A[x+i][y+j] = 0
    return A
n,m = map(int,input().split()) 
a = []
b = []
count = 0
for i in range(n):
    a.append(list(map(int,input())))
for i in range(n):
    b.append(list(map(int,input())))
for i in range(n-3+1):
    for j in range(m-3+1):
        if a[i][j] != b[i][j]:
            a = change(i,j,a)
            count += 1
answer = True
for i in range(n):
    for j in range(m):
        if a[i][j] != b[i][j]:
            answer = False
if answer:
    print(count)
else:
    print(-1)
    
