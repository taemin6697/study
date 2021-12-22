s = int(input())
n = 1
n_list = []
while n * (n + 1) / 2 <= s:
    n += 1
    n_list.append(n)
    print(n)
print(n - 1)
print(n_list)
