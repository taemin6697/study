n = int(input())
user = []
for _ in range(n):
    age,name = map(str,input().split())
    age = int(age)
    user.append((age,name))

user.sort(key = lambda user:(user[0]))

for us in user:
    print(us[0],us[1])
