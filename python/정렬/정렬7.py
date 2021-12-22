import sys
words = []
result = []
n = int(sys.stdin.readline().strip())
for _ in range(n):
    words.append(sys.stdin.readline().strip())
for value in words:
    if value not in result:
        result.append(value)
result.sort()
result.sort(key=len)
for i in result:
    print(i)
