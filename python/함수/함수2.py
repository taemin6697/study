#생성자 숫자만 뺀다
all_number = []
plus_number = []
number = []
for i in range(1,10001):
    all_number.append(i)
for i in range(1,10001):
    for j in str(i):
        i += int(j)
    plus_number.append(i)    
number = sorted(set(all_number) - set(plus_number))
for i in number:
    print(i)
        

