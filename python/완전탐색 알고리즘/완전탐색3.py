n = int(input())
student = []
for _ in range(n):
    x,y = map(int,input().split())
    student.append((x,y))

for i in student:
    rank = 1
    for j in student:
        if i[0] < j[0] and i[1] < j[1]:
            rank += 1
    print(rank,end = " ")
