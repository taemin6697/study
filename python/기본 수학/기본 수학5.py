num = int(input())
for i in range(num):
    h,w,n = map(int,input().split())
    custm=0
    j = 0
    k = 0
    for j in range(w):
        for k in range(h):
           custm+=1
           if custm >= n:
               break
        if custm >= n:
            break
    if j+1<10:
        print(k+1,0,j+1,sep='')
    else:
        print(k+1,j+1,sep='')
