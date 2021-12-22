import string
alpha = list(string.ascii_lowercase)

a = input()
for i in alpha:
        if i in a:
            print(a.index(i), end= ' ')
        else:
            print( -1, end =' ')
        
