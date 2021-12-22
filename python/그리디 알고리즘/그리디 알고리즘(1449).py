n,l = map(int,input().split())
hole = list(map(int,input().split()))#1,2,100,101
hole.sort()
cnt = 1#테이프 갯수
tmp = hole[0]#테이프 길이
for i in range(1,len(hole)):
    if tmp+l-0.5<hole[i]:
        tmp = hole[i]
        cnt +=1
print(cnt)
        
