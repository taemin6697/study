import heapq
import sys
input = sys.stdin.readline
n = int(input())
heap =[]
for i in range(n):
    heapq.heappush(heap,int(input()))
if len(heap)==1:
    print(0)
else:
    answer = 0
    while len(heap) > 1:   
        first  = heapq.heappop(heap)
        second = heapq.heappop(heap)
        answer += first + second
        heapq.heappush(heap,first+second)
    print(answer)
