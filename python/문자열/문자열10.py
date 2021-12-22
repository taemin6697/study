n = int(input())
group_word = 0
for _ in range(n):
    word = input()
    error = 0
    for index in range(len(word)-1):  # 인덱스 범위 생성 : 0부터 단어개수 -1까지
        if word[index] != word[index+1]:
            new_word = word[index+1:]#happy -> appy
            if new_word.count(word[index]) > 0: #
                error += 1
    if error ==0:
        group_word += 1
print(group_word)
