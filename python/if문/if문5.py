a,b = map(int,input().split())
if(b-45<0):
    a = a-1
    if(a<0):
        a = a+24
        print(a, b+60-45)
    else:
        print(a, b+60-45)
else:
    print(a, b-45)

    

