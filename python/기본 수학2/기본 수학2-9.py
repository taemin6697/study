while True:
    tri = list(map(int,input().split()))
    max_num = max(tri)
    if sum(tri) == 0:
        break
    tri.remove(max_num)
    if tri[0] ** 2 + tri[1] **2 == max_num ** 2:
        print("right")
    else:
        print("wrong")
