import sys
n = int(sys.stdin.readline())
b = [0]*10001
for _ in range(n):
    b[int(sys.stdin.readline())] += 1
for i in range(10001):
    for j in range(b[i]):
        print(i)
