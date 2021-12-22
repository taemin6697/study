import math
def prime(num):
    a = int(math.sqrt(num))
    count = 0
    if num == 1:
        return False
    else:
        for i in range(2,a+1):
            if num % i == 0:
                return False
        return True
num_list = list(range(2,246912))
sosu_list = []
for i in num_list:
    if prime(i):
        sosu_list.append(i)
while True:
    n = int(input())
    if n == 0:
        break
    cnt = 0
    for i in sosu_list:
        if n < i <= n*2:
            cnt += 1
    print(cnt)

