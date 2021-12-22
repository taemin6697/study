n = int(input())
number = []
for _ in range(n):
    number.append(int(input()))
nums = sorted(number)

for i in range(n):
    print(nums[i])
