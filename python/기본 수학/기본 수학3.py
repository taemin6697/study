x = int(input()) #14
line = 1
while x > line:
    x -= line #4
    line += 1 #5
if line % 2 ==0:
    a = x 
    b = line - x + 1
else:
    a = line-x+1 #5-4+1 = 2
    b = x # 4
print(a,'/',b,sep="")
