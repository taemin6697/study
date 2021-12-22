a=k=int(input())
cnt = 0
while True:
    ten = a // 10
    one = a % 10
    result = ten+one
    a = int(str(one%10)+str(result%10))
    cnt += 1
    if(a==k):
        break
print(cnt)
