def twoSum(num:list, target:int):
    mylist = []
    left = 0 
    right = len(num)-1

    while left < right:
        print(left, right)
        
        left += 1
numlist = [3,2,4]
target = 6

print(twoSum(numlist, target))