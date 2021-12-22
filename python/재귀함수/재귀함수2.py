def re(num):
    if num <= 1:
        return num
    else:
        return re(num-1) + re(num-2)

num = int(input())
print(re(num))
