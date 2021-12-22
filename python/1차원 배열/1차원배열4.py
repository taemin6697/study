number_list = []
number_div = []
for i in range(10):
    number_list.append(int(input()))
for i in number_list:
    number_div.append(i%42)
number_div = set(number_div)
print(len(number_div))

