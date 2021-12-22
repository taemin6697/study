def hansu(num):
    num_list = []
    number_list =[]
    k = []
    for i in range(0,num+1):
        if i<100:
            number_list.append(i)
        else:
            hund = (i // 100) 
            ten = (i%100)//10
            one = ((i%100)%10)
            if (hund-ten)==(ten-one): 
                number_list.append(i)
    print(len(number_list)-1)
num = int(input())
hansu(num)
