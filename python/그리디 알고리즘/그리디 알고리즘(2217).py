n = int(input())
lope = []
value = []
for _ in range(n):
    lope.append(int(input()))
result = 0
lope.sort(reverse=True)
for i in range(n):
    k = lope[i] * (i+1)
    value.append(k)
print(max(value))
    
