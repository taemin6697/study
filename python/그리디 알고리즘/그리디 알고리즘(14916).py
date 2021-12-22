money = int(input())
o = money % 5
if money ==1 or money == 3:
    print(-1)
elif o % 2 == 0:
    print(money //5 + o //2)
else:
    print((money//5)-1+(o+5)//2)
    
