import sys
input = sys.stdin.readline
n,money = map(int,input().split())
coin = []
i = 0
count = 0
for _ in range(n):
    coin.append(int(input()))
coin.sort(reverse= True)
for i in coin:
    count = count + money // i
    money %= i
print(count)
        
    
