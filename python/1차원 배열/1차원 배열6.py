num = int(input())

for i in range(num):
    a = input()
    score = 0
    sum_score = 0
    for j in a:
        if j == 'O':
            score += 1
            
        else:
            score = 0
        sum_score += score    
    print(sum_score)
