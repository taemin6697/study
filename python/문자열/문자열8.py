alpa_list = ['ABC','DEF','GHI','JKL','MNO','PQRS','TUV','WXYZ']
word = input()
time = 0
for alpa in alpa_list:
    for i in alpa:
        for x in word:
            if i == x:
                time += alpa_list.index(alpa) + 3
print(time)


