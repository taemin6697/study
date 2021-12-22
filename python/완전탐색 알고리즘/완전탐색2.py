number = int(input())
hap = []
for i in range(number+1):
    k = i
    hap.clear()
    while(i != 0):
        hap.append(i%10)
        i = i//10
    list_hap = sum(hap)
    if list_hap + k == number:
        c = 1
        break
    else:
        c = 0
if c == 1:
    print(k)
else:
    print(0)
    

