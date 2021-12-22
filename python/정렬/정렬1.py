n = int(input())
number = []
for _ in range(n):
    number.append(int(input()))
for i in range(n):
    min_number = 9999
    for j in range(i,n):    
        if min_number > number[j]:
            min_number = number[j] #min = 1 number[j] = 4
            index = j #index = 4
    temp = number[i] #temp = number[0] = 4
    number[i] = min_number#number[0] = 1
    number[index] = temp#number[4] = 4
for l in range(n):
    print(number[l])
