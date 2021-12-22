def re(num):
    if num == 1:
        return 1
    elif num == 0:
        return 1
    else:
        return num * re(num-1)
num = int(input())
print(re(num))
