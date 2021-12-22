n = int(input()) #4
street = list(map(int,input().split())) # 2 3 1
gas = list(map(int,input().split())) #   5 2 4 1
cost = gas[0]*street[0] #10
minValue = gas[0]
for i in range(1,len(street)):
    if minValue > gas[i]:
        minValue = gas[i]
    cost += minValue* street[i]
print(cost)    
