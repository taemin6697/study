
number = int(input())
for i in range(number):
    number_list = list(map(int,input().split()))
    average = sum(number_list[1:]) / int(len(number_list)-1)
    count = 0
    for j in number_list[1:]:
        if j>average:
            count += 1
    k = count / (len(number_list)-1) *100
    print("%.3f%%"%k)
    
