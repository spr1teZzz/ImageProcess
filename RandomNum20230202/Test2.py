list1 = [[0.1,0.1],[2,2],[3,3],[4,4],[5,5]]
list1 = [item for sublist in list1 for item in sublist]
print(list1)