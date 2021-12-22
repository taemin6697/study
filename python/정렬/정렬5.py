n = input()
number = []
for i in n:
    number.append(int(i))
number.sort(reverse = True)
for i in number:
    print(i,end='')
