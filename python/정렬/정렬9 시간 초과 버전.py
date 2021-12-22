import sys
input = sys.stdin.readline
n = int(input())
number =[]
append_list = []
k = []
number = list(map(int,input().split()))

for i in range(n):
    min_number = 9999
    for j in range(n):
        if number[i]>number[j]:
            append_list.append(number[j])
    result1 = set(append_list)
    result2 = list(result1)
    k.append(len(result2))

    append_list.clear()
    result1.clear()
    result2.clear()
for l in k:
    print(l,end=' ')
