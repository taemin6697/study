num = int(input())
for i in range(num):
        su, s = input().split()
        text = ""
        for i in s:
                text += int(su)*i
        print(text)
