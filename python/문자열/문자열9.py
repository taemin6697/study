alpa_list = ['c=','c-','dz=','d-','lj','nj','s=','z=']

a = input()

k=0
for t in alpa_list:
    a = a.replace(t,"*")


print(len(a))
