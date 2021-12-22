num = int(input())
score = list(map(int,input().split()))
result = 0

for i in score:
    i = i/max(score)*100
    result = result + i
result = result / int(len(score))
print(result)

