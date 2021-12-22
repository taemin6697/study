import sys
n = int(input())
number = []
for _ in range(n):
    number.append(int(sys.stdin.readline()))
print("%.f"%(sum(number)/n))

nums = sorted(number)
print(nums[n//2])

from collections import Counter
k=Counter(nums).most_common()

if len(nums) > 1:  #만약 입력값이 하나면 , 그게 최빈값이 되므로 예외처리
    if k[0][1] == k[1][1]:print(k[1][0]) 
    # 최빈값의 빈도수를 비교하여, 2개이상의 최빈값이 있으면 두번째로 작은것을 출력
    else : print(k[0][0]) 
else : print(nums[0]) 

print(nums[-1] - nums[0])
