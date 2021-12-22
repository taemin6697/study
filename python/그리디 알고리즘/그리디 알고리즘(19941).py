n,r = map(int,input().split())
HP = list(input())
k = []
count = 0
for i in range(n):
    if HP[i] =='P':
        for j in range(i-r,i+r+1):
            if 0<=j <n and HP[j] == 'H':
                count+=1
                HP[j] ="not"
                break
    else:
        continue
print(count)
    
