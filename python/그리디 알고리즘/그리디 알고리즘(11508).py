n = int(input())
data = []
for i in range(n):
    data.append(int(input()))
data.sort(reverse = True) 

ans = 0
for i in range(n):
    if i % 3 == 2: continue
    ans += data[i]

print(ans)
