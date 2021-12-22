a,b,c = map(int,input().split()) #a//(c-b)+1
if b>=c:
    print(-1)
else:
    print(a//(c-b)+1)
