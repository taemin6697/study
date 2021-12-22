number = list(input().split('-'))

result = 0
for i in number[0].split():
    result += int(i)
for k in number[1:]:
    for j in k.split('+'):
        result -= int(j)
print(result)
