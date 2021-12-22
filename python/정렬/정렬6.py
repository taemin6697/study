import sys
input = sys.stdin.readline
n = int(input())
number = []
for _ in range(n):
    [a,b] = map(int,input().split())
    number.append([b,a])

number.sort()
for i in range(n):
    print(number[i][1],number[i][0])

