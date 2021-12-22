m = int(input())
n = int(input())
sosu_list = []
sosu_sum = 0
for i in range(m,n+1):#60 61 62
    count = 0
    if i > 1:
        for j in range(2,i):
            if i % j == 0:
                count +=1
                break
        if count == 0:
            sosu_sum += i
            sosu_list.append(i)
if sosu_sum > 0:   
    print(sosu_sum)
    print(min(sosu_list))
else:
    print(-1)
