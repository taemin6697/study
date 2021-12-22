import sys
input = sys.stdin.readline
n = int(input())
number = list(map(int,input().split()))# 3 1 4 3 2
number.sort()# 1 2 3 3 4
result = 0
k = 0
for i in number:
    k += i
    result += k

print(result)
    
