import math
def isprime(num):
    if num == 1: return False
    n = int(math.sqrt(num))
    for j in range(2,n+1):
        if num % j == 0: return False
    return True

a,b = map(int,input().split())
for k in range(a,b+1):
    if isprime(k) : print(k)
