n = int(input())
numbers = map(int,input().split())
sosu = 0
for num in numbers:
    count = 0
    if num>1:
        for i in range(2, num): #i = 2,3,4,5,6
            if num % i == 0:
                count +=1
        if count == 0:
            sosu += 1
print(sosu)
