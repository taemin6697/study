a,b,v = map(int,input().split())
snail = 1
date = (v-b)/(a-b)
print(int(date) if date == int(date) else int(date)+1)
