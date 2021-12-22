n = list(input())
count = 0
change = []
change2 = []
for i in range(1,len(n)):
    if n[i-1] != n[i]:   
        change.append(i)
if len(change)%2 == 0:
    print(len(change)//2)
else:
    print((len(change)//2)+1)
