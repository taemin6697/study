import sys
first,final = map(int,sys.stdin.readline().split())
cnt = 1
while True:
    if final == first:
        break
    elif first > final or (final % 10 != 1 and final % 2 != 0):
        cnt = -1
        break
    elif final %10 == 1:
        final = final // 10
        cnt +=1
    elif final % 2 == 0:
        final = final // 2
        cnt += 1
print(cnt) 
